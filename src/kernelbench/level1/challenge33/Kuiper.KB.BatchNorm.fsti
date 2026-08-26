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

   No new [magic ()] / [admit ()] / [assume pure].  The only [magic ()]
   inherited is via [Kuiper.Kernel.Map.map_gpu]'s
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

inline_for_extraction noextract
type batchnorm_fw_ty (t:Type0) {| floating t, real_like t |} =
  fn (n  : erased nat)
     (c  : szp)
     (hw : szp { SZ.v hw > 0 /\
                 SZ.fits (n * SZ.v hw) /\
                 SZ.fits (SZ.v hw * SZ.v c) /\
                 SZ.fits (n * (SZ.v hw * SZ.v c)) })
     (nhw : szp { SZ.v nhw == n * SZ.v hw /\
                  nhw <= max_blocks * max_threads /\
                  SZ.fits (SZ.v nhw + 1024) })
     (eps : t)
     (x     : array2 t (l2_bcm_channels n (SZ.v c) (SZ.v hw))
                        { is_global x })
     (gamma : array1 t (l1_forward c) { is_global gamma })
     (beta  : array1 t (l1_forward c) { is_global beta  })
     (#fg : perm)
     (#fb : perm)
     (#sx : erased (chest2 t (SZ.v c) (n * SZ.v hw)))
     (#sg : erased (chest1 t (SZ.v c)))
     (#sb : erased (chest1 t (SZ.v c)))
     requires
       cpu **
       on gpu_loc (x |-> sx) **
       on gpu_loc (gamma |-> Frac fg sg) **
       on gpu_loc (beta  |-> Frac fb sb)
     ensures
       cpu **
       on gpu_loc (gamma |-> Frac fg sg) **
       on gpu_loc (beta  |-> Frac fb sb) **
       (exists* (sx' : chest2 t (SZ.v c) (n * SZ.v hw)).
          on gpu_loc (x |-> sx') **
          pure (batchnorm_post (SZ.v c) (n * SZ.v hw) eps (bn_inv_n nhw)
                  (chest1_to_seq sg) (chest1_to_seq sb) sx sx'))

val batchnorm_fw_f32 : batchnorm_fw_ty f32
