module Kuiper.KB.RMSNorm

(* KernelBench L1 #36: RMS normalisation along dim=1.

   Input X has shape (B, C, H, W) row-major.  For each spatial location
   (b, h, w) with hw = h*W+w:
       sum_sq[b,hw] = Σ_c X[b,c,h,w]^2
       inv_rms[b,hw] = rsqrt(sum_sq[b,hw] * inv_c + eps)
       X[b,c,h,w] ← X[b,c,h,w] * inv_rms[b,hw]

   The (B, C, H, W) row-major buffer is viewed in F* as Array2 with layout
       l2_bcm_pages b hw c
   whose imap is (r, ci) ↦ (r/HW)*C*HW + ci*HW + r%HW
   exactly matching the physical row-major layout.

   Exactly 3 GPU kernel launches:
     1. reduce_batched (x ↦ x*x)   -- sum of squares per row
     2. map_gpu (s ↦ rsqrt(s/C+ε)) -- inv_rms per row
     3. row_scale                   -- in-place scale each row

   The square-root approximation law is supplied by the temporary local
   compatibility assumption documented in patches/kuiper-sqrt-approx.md. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.RMSNorm
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT

(* The per-row reciprocal 1/C, built from the runtime [c] inside the
   verification boundary (extracts to 1.0f / (float)C) so no unverified
   floating-point arithmetic happens in the C bridge. *)
inline_for_extraction noextract
let rms_inv_c (#t:Type0) {| floating t |} (c : SZ.t) : t =
  div one (of_int (FStar.Int.Cast.uint64_to_int64
                     (FStar.SizeT.sizet_to_uint64 c)))

fn rmsnorm_fw_f32
  (b : szp)
  (hw : SZ.t { 0 < SZ.v hw /\ SZ.fits (SZ.v b * SZ.v hw) })
  (c : SZ.t { 0 < SZ.v c /\ SZ.fits (SZ.v hw * SZ.v c) /\ SZ.fits (SZ.v b * (SZ.v hw * SZ.v c)) })
  (eps : f32)
  (x : array2 f32 (l2_bcm_pages b hw c) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v hw) c)
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b * SZ.v hw > 0 /\
      SZ.v b * SZ.v hw * SZ.v c <= max_blocks * max_threads /\
      to_real eps >. 0.0R
    )
  ensures
    (exists* (sx' : chest2 f32 (SZ.v b * SZ.v hw) c).
       on gpu_loc (x |-> sx') **
       pure (rmsnorm_post (SZ.v b * SZ.v hw) c eps sx sx'))
