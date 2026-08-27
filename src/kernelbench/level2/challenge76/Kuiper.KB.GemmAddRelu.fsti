module Kuiper.KB.GemmAddRelu

(* KernelBench L2 #76: Gemm_Add_ReLU.

   PyTorch reference:
     self.gemm = nn.Linear(in, out, bias=False)   # weight (out, in)
     self.bias = nn.Parameter(randn(out,))
     x = gemm(x)          # x @ W.T            -> (batch, out)
     x = x + bias         # per-column broadcast over rows
     x = relu(x)          # max(x, 0)
   Output shape (batch, out).

   The caller passes Wᵀ as [wt] (input × out), so the linear layer is the
   plain matmul [x @ wt]; [bias] is broadcast over rows.

   Pipeline (GPU-only, fused, three launches):
     1. GEMM      : gC := x @ wt                (GEMM.Naive2.mmcomb_gpu_exact)
     2. bias-add  : y[i*out+j] := C[i,j] + bias[j]  (BiasAdd.bias_add_gpu)
     3. ReLU      : y := max(y, 0)              (Map.map_gpu relu)

   The functional postcondition is EXACT (no real-number approximation):
   each output element equals the float-level ReLU of the bias-augmented
   matmul entry.  No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Per-element EXACT functional postcondition: the flattened output at
   [(i, j)] equals the float ReLU of the bias-augmented matmul entry. *)
let gemm_add_relu_post
  (#t:Type0) {| floating t |}
  (#batch #input #out : nat)
  (sx  : EM.chest2 t batch input)
  (swt : EM.chest2 t input out)
  (sbias : chest1 t out)
  (sy' : chest1 t (batch * out))
  : prop
  = forall (i:natlt batch) (j:natlt out).
      acc1 sy' (i * out + j) ==
        relu (add (acc2 (MS.matmul sx swt) i j) (acc1 sbias j))

inline_for_extraction noextract
type gemm_add_relu_ty (t:Type0) {| floating t |} =
  fn (batch input : szp)
     (out : szp {
        SZ.v batch * SZ.v out <= max_blocks * max_threads /\
        SZ.fits (SZ.v batch * SZ.v input) /\
        SZ.fits (SZ.v input * SZ.v out) /\
        SZ.fits (SZ.v batch * SZ.v out) })
     (x    : array2 t (l2_row_major (SZ.v batch) (SZ.v input)) { is_global x    })
     (wt   : array2 t (l2_row_major (SZ.v input) (SZ.v out))   { is_global wt   })
     (bias : array1 t (l1_forward (SZ.v out))                  { is_global bias })
     (y    : array1 t (l1_forward (SZ.v batch * SZ.v out))     { is_global y    })
     (#sx   : EM.chest2 t (SZ.v batch) (SZ.v input))
     (#swt  : EM.chest2 t (SZ.v input) (SZ.v out))
     (#sbias: chest1 t (SZ.v out))
     (#sy   : chest1 t (SZ.v batch * SZ.v out))
     requires
       cpu **
       on gpu_loc (x    |-> sx)   **
       on gpu_loc (wt   |-> swt)  **
       on gpu_loc (bias |-> sbias)**
       on gpu_loc (y    |-> sy)
     ensures
       cpu **
       on gpu_loc (x    |-> sx)   **
       on gpu_loc (wt   |-> swt)  **
       on gpu_loc (bias |-> sbias)**
       (exists* (sy' : chest1 t (SZ.v batch * SZ.v out)).
          on gpu_loc (y |-> sy') **
          pure (gemm_add_relu_post sx swt sbias sy'))

val gemm_add_relu_f32 : gemm_add_relu_ty f32
