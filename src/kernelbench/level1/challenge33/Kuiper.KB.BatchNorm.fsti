module Kuiper.KB.BatchNorm

(* KernelBench L1 #33: per-channel BatchNorm (nn.BatchNorm2d, training
   forward), in place.

   Physical input: float32 (N, C, H, W) row-major.  We view it as the
   strided 2-D matrix (C, N*HW) via [Kuiper.Tensor.Layout.BCMChannels]
   so that "row ci" of the matrix is the entirety of channel ci across
   all batch and spatial positions.  For each channel ci:
       sum_ci   = Σ_k x[ci,k]
       sumsq_ci = Σ_k x[ci,k]^2
       mean_ci  = sum_ci   * inv_n      -- inv_n ≈ 1/(N*H*W)
       m2_ci    = sumsq_ci * inv_n
       var_ci   = m2_ci - mean_ci^2     -- biased variance
       inv_ci   = 1 / sqrt(var_ci + eps)
       y[ci,k]  = (x[ci,k] - mean_ci) * inv_ci * γ[ci] + β[ci]

   Composes verified Kuiper primitives:
     - Kuiper.Array2.{extract,restore}_row_on_gpu_loc
                                     (host-side per-channel row split)
     - Kuiper.Kernel.HReduce.reduce  (sum / sumsq with pre_map)
     - Kuiper.Kernel.Map.map_gpu      (two-stage affine_step: (inv,-mean*inv)
                                      then (γ_ci, β_ci))
     - Kuiper.Array1.arr_read_1      (host read of γ[ci], β[ci])

   The real-valued proof uses the temporary
   [Kuiper.KB.Compat.RsqrtApprox.rsqrt_approx] assumption.  The selected
   package provides the corresponding [sqrt_approx] law but not yet this
   reciprocal-square-root law.  There is
   no challenge-local [magic ()], [admit ()], or [assume pure].  The [magic]
   inherited via [Kuiper.Kernel.Map.map_gpu]'s
   [kpre_sendable]/[kpost_sendable] (tree-wide debt for plain
   kernel_desc kernels). *)

#lang-pulse
open Kuiper
open Kuiper.Float.Casts
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMChannels
open Kuiper.EMatrix
open Kuiper.Spec.BatchNorm
module SZ = Kuiper.SizeT

(* The per-channel reciprocal 1/(N*H*W), built from the runtime [nhw]
   inside the verification boundary (extracts to 1.0f / (float)nhw).
   Defined here so the kernel and its postcondition share one value. *)
inline_for_extraction noextract
let bn_inv_n (#t:Type0) {| floating t |} (nhw : szp) : t =
  div one (of_int (FStar.Int.Cast.uint64_to_int64
                     (FStar.SizeT.sizet_to_uint64 nhw)))

fn batchnorm_fw_f32
  (n  : erased nat)
  (c  : szp)
  (#hw_n : erased pos)
  (hw : szp { SZ.v hw == reveal hw_n /\
              SZ.fits (n * (reveal hw_n)) /\
              SZ.fits ((reveal hw_n) * c) /\
              SZ.fits (n * ((reveal hw_n) * c)) })
  (nhw : szp { SZ.v nhw == n * (reveal hw_n) /\
               nhw <= max_blocks * max_threads /\
               SZ.fits (SZ.v nhw + 1024) })
  (eps : f32)
  (x     : array2 f32 (l2_bcm_channels n c (reveal hw_n))
                     { is_global x })
  (gamma : array1 f32 (l1_forward c) { is_global gamma })
  (beta  : array1 f32 (l1_forward c) { is_global beta  })
  (#fg #fb : perm)
  (#sx : chest2 f32 c (n * (reveal hw_n)))
  (#sg #sb : chest1 f32 c)
  (reps : erased real)
  (rx : erased (v : chest2 real c (n * (reveal hw_n)) {
    batchnorm_domain c (n * (reveal hw_n)) (reveal reps)
      (bn_inv_n_r (n * (reveal hw_n))) v }))
  (rg rb : erased (lseq real c))
  (#ax : squash (sx %~ ((reveal rx) <: chest2 real c
    (n * (reveal hw_n)))))
  norewrite
  preserves
    cpu **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sb) **
    pure (eps %~ reveal reps) **
    pure (sg %~ seq_to_chest1 (reveal rg)) **
    pure (sb %~ seq_to_chest1 (reveal rb))
  requires
    on gpu_loc (x |-> sx)
  ensures
    (exists* (sx' : chest2 f32 c (n * (reveal hw_n))).
       on gpu_loc (x |-> sx') **
       pure (batchnorm_post c (n * (reveal hw_n)) (reveal reps)
               (bn_inv_n_r (n * (reveal hw_n))) (reveal rg) (reveal rb)
               (reveal rx) sx'))

(* Out-of-place NCHW entry.  H*W, N*H*W, and the pybind f64-to-f32 epsilon
   conversion are performed inside Kuiper; the original input and affine
   parameters are preserved. *)
fn batchnorm2d_alloc_f32
  (n c h w : szp)
  (eps : f64)
  (x : array2 f32 (l2_bcm_channels (SZ.v n) c (h * w)) { is_global x })
  (gamma : array1 f32 (l1_forward c) { is_global gamma })
  (beta : array1 f32 (l1_forward c) { is_global beta })
  (#fx #fg #fb : perm)
  (#sx : chest2 f32 c (SZ.v n * (SZ.v h * SZ.v w)))
  (#sg #sb : chest1 f32 c)
  (rx : erased (v : chest2 real c (SZ.v n * (SZ.v h * SZ.v w)) {
    batchnorm_domain c (SZ.v n * (SZ.v h * SZ.v w))
      (to_real (fcast #f64 #f32 eps))
      (bn_inv_n_r (SZ.v n * (SZ.v h * SZ.v w))) v }))
  (rg rb : erased (lseq real c))
  preserves
    cpu ** on gpu_loc (x |-> Frac fx sx) **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta |-> Frac fb sb) **
    pure (sx %~ ((reveal rx) <: chest2 real c
            (SZ.v n * (SZ.v h * SZ.v w)))) **
    pure (sg %~ seq_to_chest1 (reveal rg)) **
    pure (sb %~ seq_to_chest1 (reveal rb))
  requires
    pure (SZ.v h > 0 /\ SZ.v w > 0 /\
          SZ.fits (SZ.v h * SZ.v w) /\
          SZ.fits (SZ.v n * (SZ.v h * SZ.v w)) /\
          SZ.fits ((SZ.v h * SZ.v w) * SZ.v c) /\
          SZ.fits (SZ.v n * ((SZ.v h * SZ.v w) * SZ.v c)) /\
          SZ.v n * (SZ.v h * SZ.v w) <= max_blocks * max_threads /\
          SZ.fits (SZ.v n * (SZ.v h * SZ.v w) + 1024))
  returns out : array2 f32
    (l2_bcm_channels (SZ.v n) c (h * w))
  ensures
    exists* (sx' : chest2 f32 c (SZ.v n * (SZ.v h * SZ.v w))).
      on gpu_loc (out |-> sx') **
      pure (batchnorm_post c (SZ.v n * (SZ.v h * SZ.v w))
              (to_real (fcast #f64 #f32 eps))
              (bn_inv_n_r (SZ.v n * (SZ.v h * SZ.v w)))
              (reveal rg) (reveal rb) (reveal rx) sx')
