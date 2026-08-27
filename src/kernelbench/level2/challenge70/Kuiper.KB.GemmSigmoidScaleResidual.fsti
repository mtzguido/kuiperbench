module Kuiper.KB.GemmSigmoidScaleResidual

(* KernelBench L2 #70: Gemm_Sigmoid_Scaling_ResidualAdd.

   PyTorch reference:
     self.gemm = nn.Linear(in, out)   # weight (out, in), WITH bias
     x = gemm(x)                       # x @ W.T + bias    -> (batch, out)
     original_x = x
     x = sigmoid(x)
     x = x * scaling_factor
     x = x + original_x                # residual add
   Output shape (batch, out).

   The caller passes Wᵀ as [wt] (input × out), so the linear layer is the
   plain matmul [x @ wt]; [bias] is broadcast over rows.  Let
     g = (x @ wt)[i,j] + bias[j]
   then the output is
     sigmoid(g) * scaling_factor + g.

   Pipeline (GPU-only, fused, three launches):
     1. GEMM      : gC := x @ wt                (GEMM.Naive2.mmcomb_gpu_exact)
     2. bias-add  : y[i*out+j] := C[i,j] + bias[j]  (BiasAdd.bias_add_gpu)
     3. fused map : y := sigmoid(y)*sf + y      (Map.map_gpu (sig_scale_res sf))

   The functional postcondition is EXACT (no real-number approximation):
   each output element equals the float-level [sig_scale_res] of the
   bias-augmented matmul entry.  No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Float-level sigmoid over the abstract floating typeclass: 1/(1+exp(-v)). *)
inline_for_extraction noextract
let sigmoid (#t:Type0) {| floating t |} (v : t) : t =
  div one (add one (fexp (sub zero v)))

(* The fused per-element step: sigmoid(v)*sf + v  (exactly torch's order). *)
inline_for_extraction noextract
let sig_scale_res (#t:Type0) {| floating t |} (sf : t) (v : t) : t =
  add (mul (sigmoid v) sf) v

(* Per-element EXACT functional postcondition: the flattened output at
   [(i, j)] equals the float [sig_scale_res] of the bias-augmented matmul
   entry [g = matmul[i,j] + bias[j]]. *)
let gemm_sigmoid_scale_residual_post
  (#t:Type0) {| floating t |}
  (#batch #input #out : nat)
  (sf : t)
  (sx  : EM.chest2 t batch input)
  (swt : EM.chest2 t input out)
  (sbias : chest1 t out)
  (sy' : chest1 t (batch * out))
  : prop
  = forall (i:natlt batch) (j:natlt out).
      acc1 sy' (i * out + j) ==
        sig_scale_res sf (add (acc2 (MS.matmul sx swt) i j) (acc1 sbias j))

inline_for_extraction noextract
type gemm_sigmoid_scale_residual_ty (t:Type0) {| floating t |} =
  fn (batch input : szp)
     (out : szp {
        SZ.v batch * SZ.v out <= max_blocks * max_threads /\
        SZ.fits (SZ.v batch * SZ.v input) /\
        SZ.fits (SZ.v input * SZ.v out) /\
        SZ.fits (SZ.v batch * SZ.v out) })
     (sf : t)
     (x    : array2 t (l2_row_major batch input) { is_global x    })
     (wt   : array2 t (l2_row_major input out)   { is_global wt   })
     (bias : array1 t (l1_forward out)                  { is_global bias })
     (y    : array1 t (l1_forward (SZ.v batch * SZ.v out))     { is_global y    })
     (#sx   : EM.chest2 t batch input)
     (#swt  : EM.chest2 t input out)
     (#sbias: chest1 t out)
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
          pure (gemm_sigmoid_scale_residual_post sf sx swt sbias sy'))

val gemm_sigmoid_scale_residual_f32 : gemm_sigmoid_scale_residual_ty f32
