module Kuiper.KB.MatmulScaleResidual

(* KernelBench L2 #40: Matmul_Scaling_ResidualAdd.

   PyTorch reference (nn.Linear, with bias):
     x = matmul(x)                  # x @ W.T + bias   -> (batch, out)
     original_x = x.clone()
     x = x * scaling_factor
     x = x + original_x             # residual add
   Output shape (batch, out).  Equivalently each element is
     g * scaling_factor + g   where  g = (x @ W.T + bias).

   The caller passes Wᵀ as [wt] (input × out), so the linear layer is the
   plain matmul [x @ wt] plus a per-column [bias] broadcast over rows.

   Pipeline (GPU-only, fused, three launches):
     1. GEMM      : gC := x @ wt                (Kuiper.Kernel.GEMM.Naive2.mmcomb_gpu_exact)
     2. bias-add  : y[i*out+j] := C[i,j] + bias[j]  (Kuiper.Kernel.BiasAdd.bias_add_gpu)
     3. scale+res : y := y * sf + y             (Kuiper.Kernel.Map.map_gpu)

   The functional postcondition is EXACT (no real-number approximation): each
   output element equals the float-level [scale_residual] of the bias-augmented
   matmul entry.  We write the closure EXACTLY as PyTorch computes it
   ([v * sf + v], NOT pre-simplified to [v * (1 + sf)]) to stay bit-faithful.

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Fused scale-and-residual step, written EXACTLY as PyTorch computes it:
   [v * sf + v] (NOT pre-simplified to [v * (1 + sf)]) to stay bit-faithful.
   A named closure so the [map_gpu] lambda and the spec-side reasoning refer
   to the same function (avoids anonymous-lambda mismatch). *)
inline_for_extraction noextract
let scale_residual (#t:Type0) {| scalar t |} (sf v : t) : t = add (mul v sf) v

(* Per-element EXACT functional postcondition: the flattened output at
   [(i, j)] equals the float [scale_residual] of the bias-augmented matmul
   entry.  Here [add], [mul], [acc2] come from [scalar]. *)
let matmul_scale_residual_post
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
        scale_residual sf (add (acc2 (MS.matmul sx swt) i j) (acc1 sbias j))

inline_for_extraction noextract
type matmul_scale_residual_ty (t:Type0) {| floating t |} =
  fn (batch input : szp)
     (out : szp {
        SZ.v batch * SZ.v out <= max_blocks * max_threads /\
        SZ.fits (SZ.v batch * SZ.v input) /\
        SZ.fits (SZ.v input * SZ.v out) /\
        SZ.fits (SZ.v batch * SZ.v out) })
     (sf : t)
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
          pure (matmul_scale_residual_post sf sx swt sbias sy'))

val matmul_scale_residual_f32 : matmul_scale_residual_ty f32
