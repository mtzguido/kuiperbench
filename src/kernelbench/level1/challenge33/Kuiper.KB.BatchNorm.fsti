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
   [Kuiper.KB.Compat.SqrtApprox.rsqrt_approx] assumption, whose upstream
   Kuiper patch is tracked in [patches/kuiper-sqrt-approx.patch].  There is
   no challenge-local [magic ()], [admit ()], or [assume pure].  The [magic]
   inherited via [Kuiper.Kernel.Map.map_gpu]'s
   [kpre_sendable]/[kpost_sendable] (tree-wide debt for plain
   kernel_desc kernels). *)

#lang-pulse
open Kuiper
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
  (hw : szp { SZ.v hw > 0 /\
              SZ.fits (n * SZ.v hw) /\
              SZ.fits (SZ.v hw * c) /\
              SZ.fits (n * (SZ.v hw * c)) })
  (nhw : szp { SZ.v nhw == n * SZ.v hw /\
               nhw <= max_blocks * max_threads /\
               SZ.fits (SZ.v nhw + 1024) })
  (eps : f32)
  (x     : array2 f32 (l2_bcm_channels n c hw)
                     { is_global x })
  (gamma : array1 f32 (l1_forward c) { is_global gamma })
  (beta  : array1 f32 (l1_forward c) { is_global beta  })
  (#fg #fb : perm)
  (#sx : chest2 f32 c (n * SZ.v hw))
  (#sg #sb : chest1 f32 c)
  (reps : erased real)
  (rx : erased (v : chest2 real c (n * SZ.v hw) {
    batchnorm_domain c (n * SZ.v hw) (reveal reps)
      (bn_inv_n_r (SZ.v nhw)) v }))
  (rg rb : erased (lseq real c))
  preserves
    cpu **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sb) **
    pure (eps %~ reveal reps /\
          sx %~ ((reveal rx) <: chest2 real c (n * SZ.v hw)) /\
          sg %~ seq_to_chest1 (reveal rg) /\
          sb %~ seq_to_chest1 (reveal rb))
  requires
    on gpu_loc (x |-> sx)
  ensures
    (exists* (sx' : chest2 f32 c (n * SZ.v hw)).
       on gpu_loc (x |-> sx') **
       pure (batchnorm_post c (n * SZ.v hw) (reveal reps)
               (bn_inv_n_r (SZ.v nhw)) (reveal rg) (reveal rb)
               (reveal rx) sx'))
