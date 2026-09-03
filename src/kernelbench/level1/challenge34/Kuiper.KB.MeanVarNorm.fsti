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
   identity affine).  Their public Kuiper entries derive the appropriate
   [(b, d)] row geometry from raw NCHW/group dimensions.

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
open Kuiper.Float.Casts
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

(* Complete KernelBench-facing entries.  Both preserve the original input,
   convert the pybind f64 epsilon to f32, allocate their result inside Kuiper,
   copy the input under the verified surface, and normalize the copy.  The
   4-D geometry is intentionally kept raw at the ABI: all row geometry,
   including GroupNorm's [c / groups], is derived below the verification
   boundary. *)
fn instancenorm34_alloc_f32
  (b c h w : szp)
  (eps : f64)
  (x : array1 f32 (l1_forward ((b * c) * (h * w))) { is_global x })
  (#sx : chest1 f32 ((b * c) * (h * w)))
  (#fx : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (x |-> Frac fx sx)
  requires
    pure (SZ.fits (b * c) /\
          SZ.fits (h * w) /\
          SZ.fits ((b * c) * (h * w)) /\
          h * w <= max_blocks * max_threads /\
          (b * c) * (h * w) <= max_blocks * max_threads /\
          mean_var_domain (b * c) (h * w) (fcast #f64 #f32 eps)
            (chest1_to_seq sx))
  returns out : array1 f32 (l1_forward ((b * c) * (h * w)))
  ensures
    exists* (sout : chest1 f32 ((b * c) * (h * w))).
      on gpu_loc (out |-> sout) **
      pure (mean_var_post (b * c) (h * w) (fcast #f64 #f32 eps)
              (chest1_to_seq sx) (chest1_to_seq sout))

fn groupnorm35_alloc_f32
  (b c h w : szp)
  (groups : szp { groups <= c /\ c / groups > 0 })
  (eps : f64)
  (x : array1 f32
         (l1_forward ((b * groups) * ((c / groups) * (h * w))))
       { is_global x })
  (#sx : chest1 f32 ((b * groups) * ((c / groups) * (h * w))))
  (#fx : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (x |-> Frac fx sx)
  requires
    pure (groups <= c /\ c % groups == 0 /\
          SZ.fits (b * groups) /\
          SZ.fits (h * w) /\
          SZ.fits ((c / groups) * (h * w)) /\
          SZ.fits ((b * groups) * ((c / groups) * (h * w))) /\
          (c / groups) * (h * w) <= max_blocks * max_threads /\
          (b * groups) * ((c / groups) * (h * w)) <=
            max_blocks * max_threads /\
          mean_var_domain (b * groups) ((c / groups) * (h * w))
            (fcast #f64 #f32 eps)
            (chest1_to_seq sx))
  returns out : array1 f32
    (l1_forward ((b * groups) * ((c / groups) * (h * w))))
  ensures
    exists* (sout : chest1 f32
      ((b * groups) * ((c / groups) * (h * w)))).
      on gpu_loc (out |-> sout) **
      pure (mean_var_post (b * groups) ((c / groups) * (h * w))
              (fcast #f64 #f32 eps)
              (chest1_to_seq sx) (chest1_to_seq sout))
