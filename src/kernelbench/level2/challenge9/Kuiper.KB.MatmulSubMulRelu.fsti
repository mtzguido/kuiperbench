module Kuiper.KB.MatmulSubMulRelu

(* KernelBench L2 #9: Matmul_Subtract_Multiply_ReLU.

   PyTorch reference (nn.Linear, with bias):
     x = linear(x)            # x @ W.T + bias   -> (batch, out)
     x = x - subtract_value   # subtract a constant
     x = x * multiply_value   # multiply by a constant
     x = relu(x)              # max(x, 0)
   Output shape (batch, out).

   The caller passes Wᵀ as [wt] (input × out), so the linear layer is the
   plain matmul [x @ wt] plus a per-column [bias] broadcast over rows.

   Pipeline (GPU-only, fused, three launches):
     1. GEMM      : gC := x @ wt                    (Kuiper.Kernel.GEMM.Naive2.mmcomb_gpu_exact)
     2. bias-add  : y[i*out+j] := C[i,j] + bias[j]  (Kuiper.Kernel.BiasAdd.bias_add_gpu)
     3. fused map : y := relu((y - sub_v) * mul_v)  (Kuiper.Kernel.Map.map_gpu)

   The functional postcondition is EXACT (no real-number approximation): each
   output element equals the float-level composition of the steps.

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Per-element EXACT functional postcondition: the flattened output at
   [(i, j)] equals the float-level [relu (mul (sub (gemm+bias) sub_v) mul_v)]
   of the bias-augmented matmul entry.  Here [sub], [mul], [relu] are the
   float operations of the [floating] typeclass, [add] / [acc2] come from
   [scalar]. *)
let matmul_sub_mul_relu_post
  (#t:Type0) {| floating t |}
  (#batch #input #out : nat)
  (sub_v mul_v : t)
  (sx  : EM.chest2 t batch input)
  (swt : EM.chest2 t input out)
  (sbias : chest1 t out)
  (sy' : chest1 t (batch * out))
  : prop
  = forall (i:natlt batch) (j:natlt out).
      acc1 sy' (i * out + j) ==
        relu (mul (sub (add (acc2 (MS.matmul sx swt) i j) (acc1 sbias j)) sub_v) mul_v)

inline_for_extraction noextract
type matmul_sub_mul_relu_ty (t:Type0) {| floating t |} =
  fn (batch : szp)
     (input : szp)
     (out : szp {
        SZ.v batch * SZ.v out <= max_blocks * max_threads /\
        SZ.fits (SZ.v batch * SZ.v input) /\
        SZ.fits (SZ.v input * SZ.v out) /\
        SZ.fits (SZ.v batch * SZ.v out) })
     (sub_v : t)
     (mul_v : t)
     (x    : array2 t (l2_row_major (SZ.v batch) (SZ.v input)) { is_global x    })
     (wt   : array2 t (l2_row_major (SZ.v input) (SZ.v out))   { is_global wt   })
     (bias : array1 t (l1_forward (SZ.v out))                  { is_global bias })
     (y    : array1 t (l1_forward (SZ.v batch * SZ.v out))     { is_global y    })
     (#sx   : erased (EM.chest2 t (SZ.v batch) (SZ.v input)))
     (#swt  : erased (EM.chest2 t (SZ.v input) (SZ.v out)))
     (#sbias: erased (chest1 t (SZ.v out)))
     (#sy   : erased (chest1 t (SZ.v batch * SZ.v out)))
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
          pure (matmul_sub_mul_relu_post sub_v mul_v sx swt sbias sy'))

val matmul_sub_mul_relu_f32 : matmul_sub_mul_relu_ty f32
