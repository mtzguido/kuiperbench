module Kuiper.KB.MaxPool2D

(* KernelBench L1 #42: MaxPool2D.
 *
 * 2-D max pooling reduces to two passes of the verified
 * [Kuiper.Kernel.WindowReduce1D] primitive instantiated with
 * [reducer_fmax_f32] (rid = -inf, rop = fmaxf):
 *
 *   pass 1: per-row max over W axis (B*C*H rows of length W)
 *   pass 2: per-row max over H axis (B*C*W_out rows of length H), using a
 *           verified zero-copy BCM-pages layout recast
 *
 * Each pass has an exact implementation-order left-fold specification.
 * The public raw entry owns the complete two-pass composition, including
 * allocation, the layout recast, and cleanup.  The C++ bridge only checks
 * the raw dimension contract and calls it once; no [fmax] algebra is assumed.
 *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout { from_seq, to_seq }
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { reducer_fmax_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
open Kuiper.Tensor.Layout.BCMPages { l2_bcm_pages }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix

noeq type maxpool2d_full_result
  (kh kw sh sw ph pw dh dw bc h w : szp) = {
  w_out : wo:sz { SZ.v wo == pool_out_len_1d w kw sw pw dw /\
                  SZ.v wo > 0 };
  h_out : ho:sz { SZ.v ho == pool_out_len_1d h kh sh ph dh /\
                  SZ.v ho > 0 };
  output : array2 f32 (l2_bcm_pages bc w_out h_out);
}

(* Verified, extractable 1-D pool output-length formula (see .fst).  Provably
   equal to the pure spec [pool_out_len_1d]; the C bridge calls this (per axis)
   instead of re-implementing the formula in unverified C. *)
val pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)

(* The exact ghost value presented to the height pass after the width-pass
   result is reinterpreted through the BCM-pages layout.  The recast moves no
   data: [to_seq] exposes the width-pass buffer in physical order and
   [from_seq] views those same elements as (bc * w_out, h). *)
unfold
let maxpool2d_mid_view
  (bc h w : nat)
  (kw sw : pos) (pw : nat) (dw : pos)
  (w_out : pos)
  (sx : chest2 f32 (bc * h) w)
  : chest2 f32 (bc * w_out) h
  = from_seq (l2_bcm_pages bc w_out h)
      (to_seq (l2_row_major (bc * h) w_out)
        (windowreduce_result reducer_fmax_f32 sx
          kw sw pw dw w_out))

(* Verification-facing wrapper type (layout-polymorphic, f32 carrier). *)
inline_for_extraction noextract
fn maxpool2d_axis_fw_f32
  (k s p d : szp)
(bc : szp { SZ.v bc <= max_blocks * max_threads })
(l    : szp)
(l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
(#lin  : layout2 bc l) {| ctlayout lin  |}
(#lout : layout2 bc l_out) {| ctlayout lout |}
(input  : array2 f32 lin  { is_global input  })
(output : array2 f32 lout { is_global output })
(#fIn  : perm)
(#sx   : chest2 f32 bc l)
(#sout : chest2 f32 bc l_out)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 on gpu_loc (output |-> sout) **
 pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
 pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
ensures
 on gpu_loc (output |->
   windowreduce_result reducer_fmax_f32 sx
     k s p d l_out)


(* Concrete-layout extractable entry (l2_row_major). *)
fn maxpool2d_axis_fw_rm_f32
  (k s p d : szp)
(bc : szp { SZ.v bc <= max_blocks * max_threads })
(l    : szp { SZ.fits (SZ.v bc * SZ.v l) })
(l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d /\
             SZ.fits (SZ.v bc * SZ.v l_out) })
(input  : array2 f32 (l2_row_major bc l)     { is_global input  })
(output : array2 f32 (l2_row_major bc l_out) { is_global output })
(#fIn  : perm)
(#sx   : chest2 f32 bc l)
(#sout : chest2 f32 bc l_out)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 on gpu_loc (output |-> sout) **
 pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
 pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
ensures
 on gpu_loc (output |->
   windowreduce_result reducer_fmax_f32 sx
     k s p d l_out)


(* ── Single verified, transpose-free 2-D max-pool entry ──────────────

   [maxpool2d_full_alloc_f32] folds the WHOLE separable 2-D max pool into
   one verified F*/Pulse call, eliminating the unverified PyTorch
   [.permute().contiguous()] that used to sit between the two
   [windowreduce] passes in the C++ bridge.

   Input is a row-major (B*C*H, W) view ([bc = B*C], inner = W).  The
   returned record's buffer is physically the
   row-major (B*C, H_out, W_out) result.  Its post composes both passes:
   [maxpool2d_mid_view] is exactly the width-pass result viewed as
   (B*C*W_out, H), and the returned buffer is its height-pass result. *)
fn maxpool2d_full_alloc_f32
  (kh kw sh sw ph pw dh dw bc h w : szp)
(#_ : squash (SZ.fits (SZ.v bc * SZ.v h)))
(input : array2 f32 (l2_row_major (bc * h) w) { is_global input })
(#fIn : perm)
(#sx  : chest2 f32 (bc * h) w)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 pure (SZ.fits (SZ.v dw * (SZ.v kw - 1) + 1)) **
 pure (SZ.fits (SZ.v dh * (SZ.v kh - 1) + 1)) **
 pure (SZ.fits (SZ.v w + 2 * SZ.v pw)) **
 pure (SZ.fits (SZ.v h + 2 * SZ.v ph)) **
 pure (SZ.v dw * (SZ.v kw - 1) + 1 <= SZ.v w + 2 * SZ.v pw) **
 pure (SZ.v dh * (SZ.v kh - 1) + 1 <= SZ.v h + 2 * SZ.v ph) **
 pure (SZ.fits ((SZ.v w + 2 * SZ.v pw) * SZ.v sw + SZ.v kw * SZ.v dw)) **
 pure (SZ.fits ((SZ.v h + 2 * SZ.v ph) * SZ.v sh + SZ.v kh * SZ.v dh)) **
 pure (SZ.fits (SZ.v bc * (SZ.v h + 2 * SZ.v ph) * (SZ.v w + 2 * SZ.v pw))) **
 pure (SZ.v bc * (SZ.v h + 2 * SZ.v ph) * (SZ.v w + 2 * SZ.v pw)
         <= max_blocks * max_threads)
returns r : maxpool2d_full_result kh kw sh sw ph pw dh dw bc h w
ensures
 on gpu_loc (r.output |->
   windowreduce_result reducer_fmax_f32
     (maxpool2d_mid_view bc h w kw sw pw dw r.w_out sx)
     kh sh ph dh r.h_out)

(* KernelBench-shaped entry: derives [bc = b*c] and duplicates the scalar
   pooling parameters across H/W inside Kuiper. *)
fn maxpool2d_raw_alloc_f32
  (k s p d b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) })
  (h w : szp)
  (#_ : squash (SZ.fits (SZ.v (b *^ c) * SZ.v h)))
  (input : array2 f32 (l2_row_major ((b *^ c) * h) w) { is_global input })
  (#fIn : perm)
  (#sx : chest2 f32 ((b *^ c) * h) w)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
    pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
    pure (SZ.fits (SZ.v w + 2 * SZ.v p)) **
    pure (SZ.fits (SZ.v h + 2 * SZ.v p)) **
    pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v w + 2 * SZ.v p) **
    pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v h + 2 * SZ.v p) **
    pure (SZ.fits ((SZ.v w + 2 * SZ.v p) * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.fits ((SZ.v h + 2 * SZ.v p) * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.fits (SZ.v (b *^ c) * (SZ.v h + 2 * SZ.v p)
                    * (SZ.v w + 2 * SZ.v p))) **
    pure (SZ.v (b *^ c) * (SZ.v h + 2 * SZ.v p)
            * (SZ.v w + 2 * SZ.v p) <= max_blocks * max_threads)
  returns r : maxpool2d_full_result k k s s p p d d (b *^ c) h w
  ensures
    on gpu_loc (r.output |->
      windowreduce_result reducer_fmax_f32
        (maxpool2d_mid_view (b *^ c) h w k s p d r.w_out sx)
        k s p d r.h_out)
