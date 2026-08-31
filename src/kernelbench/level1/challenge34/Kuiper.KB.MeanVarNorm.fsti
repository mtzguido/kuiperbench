module Kuiper.KB.MeanVarNorm

(* KernelBench helper: row-wise mean/variance normalisation, in place.
   Treats the input as a flat (b * d) array, viewed as [b] rows of [d].
   For each row r, computes
       mean = (1/d) * Σ_j  x[r,j]
       var  = (1/d) * Σ_j  x[r,j]^2  - mean^2
       inv  = 1 / sqrt(var + eps)
       x[r,:] ← (x[r,:] - mean) * inv
              = inv * x[r,:] + (-mean*inv)

   Used by KernelBench L1 #34 (InstanceNorm) and #35 (GroupNorm with
   identity affine), with appropriate (b, d) reshapes performed in the
   bridge.

   Composes verified Kuiper primitives:
     - Kuiper.Scalars.square         (pointwise square)
     - Kuiper.Kernel.HReduce.reduce (sum)
     - Kuiper.Spec.Frobenius.affine_step (apply (inv, -mean*inv) via map_gpu)
   Per row uses the device-to-device offset memcpy primitive to copy a
   row in/out of a fixed-size scratch buffer.  Its direct real proof uses
   the temporary [rsqrt_approx] compatibility assumption documented in the
   repository patch. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.MeanVarNorm
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let mvn_inv_d (#t:Type0) {| floating t |} (d : szp) : t =
  div one (of_int (FStar.Int.Cast.uint64_to_int64
                     (FStar.SizeT.sizet_to_uint64 d)))

inline_for_extraction noextract
type mean_var_norm_fw_ty (t:Type0)
  {| floating t, real_like t, floating_real_like t |} =
  fn (b : szp)
     (d : szp { d <= max_blocks * max_threads /\
                SZ.fits (b * d) /\
                b * d <= max_blocks * max_threads })
     (eps : t)
     (x : array1 t (l1_forward (b * d)) { is_global x })
     (#s : chest1 t (b * d))
     preserves cpu
     requires on gpu_loc (x |-> s) **
       pure (mean_var_domain b d eps (chest1_to_seq s))
     ensures
       (exists* (s' : chest1 t (b * d)).
          on gpu_loc (x |-> s') **
          pure (mean_var_post b d eps
                  (chest1_to_seq s) (chest1_to_seq s')))

val mean_var_norm_fw_f32 : mean_var_norm_fw_ty f32
