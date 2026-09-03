module Kuiper.KB.MaxPool3D

(* KernelBench L1 #43: MaxPool3D.
 *
 * 3-D max pooling reduces to three passes of the verified
 * [Kuiper.Kernel.WindowReduce1D] primitive instantiated with
 * [reducer_fmax_f32] (rid = -inf, rop = fmaxf):
 *
 *   pass 1: per-row max over W axis (B*C*D*H rows of length W)
 *   pass 2: per-row max over H through a verified strided BCM layout
 *   pass 3: per-row max over D through a second verified BCM layout
 *
 * The public [maxpool3d_raw_alloc_f32] entry owns the complete composition:
 * it allocates intermediates, runs all three kernels, performs zero-copy
 * verified layout recasts between them, frees intermediates, and returns the
 * final row-major buffer.  Its postcondition is the exact nested sequence of
 * implementation-order [fmax] folds tied directly to the original input.
 * The C++ bridge performs checks and invokes this entry exactly once.
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

(* Private composition helpers have declarations here only to give their
   extraction wrappers precise expected types; the bridge calls only the raw
   entry below. *)
inline_for_extraction noextract
fn maxpool3d_axis_fw_f32
  (k s p d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
  (#lin : layout2 bc l) {| ctlayout lin |}
  (#lout : layout2 bc l_out) {| ctlayout lout |}
  (input : array2 f32 lin { is_global input })
  (output : array2 f32 lout { is_global output })
  (#fIn : perm)
  (#sx : chest2 f32 bc l)
  (#sout : chest2 f32 bc l_out)
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
  ensures
    on gpu_loc (output |->
      windowreduce_result reducer_fmax_f32 sx k s p d l_out)

fn maxpool3d_axis_fw_rm_f32
  (k s p d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l : szp { SZ.fits (SZ.v bc * SZ.v l) })
  (l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d /\
                SZ.fits (SZ.v bc * SZ.v l_out) })
  (input : array2 f32 (l2_row_major bc l) { is_global input })
  (output : array2 f32 (l2_row_major bc l_out) { is_global output })
  (#fIn : perm)
  (#sx : chest2 f32 bc l)
  (#sout : chest2 f32 bc l_out)
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
  ensures
    on gpu_loc (output |->
      windowreduce_result reducer_fmax_f32 sx k s p d l_out)

(* Exact ghost view after the W pass, reinterpreted without copying so H is
   the inner reduction axis.  Physically the bytes remain row-major
   (bc,D,H,W_out). *)
unfold
let maxpool3d_mid_w_view
  (bc depth h w : nat)
  (kw sw : pos) (pw : nat) (dw : pos)
  (w_out : pos)
  (sx : chest2 f32 (bc * depth * h) w)
  : chest2 f32 (bc * depth * w_out) h
  = from_seq (l2_bcm_pages (bc * depth) w_out h)
      (to_seq (l2_row_major (bc * depth * h) w_out)
        (windowreduce_result reducer_fmax_f32 sx
          kw sw pw dw w_out))

(* Exact ghost view after the H pass, reinterpreted without copying so D is
   the inner reduction axis.  Physically the bytes are row-major
   (bc,D,H_out,W_out). *)
unfold
let maxpool3d_mid_h_view
  (bc depth h w : nat)
  (kh kw sh sw : pos) (ph pw : nat) (dh dw : pos)
  (w_out h_out : pos)
  (sx : chest2 f32 (bc * depth * h) w)
  : chest2 f32 (bc * (h_out * w_out)) depth
  = from_seq (l2_bcm_pages bc (h_out * w_out) depth)
      (to_seq (l2_bcm_pages (bc * depth) w_out h_out)
        (windowreduce_result reducer_fmax_f32
          (maxpool3d_mid_w_view bc depth h w kw sw pw dw w_out sx)
          kh sh ph dh h_out))

(* Host validation establishes these arithmetic side conditions as one pure
   fact.  Keeping them in a named predicate prevents Pulse's separating-
   conjunction elaborator from duplicating the large dependent layout context
   once per condition. *)
unfold
let maxpool3d_full_pre
  (kd kh kw sd sh sw : pos) (pd ph pw : nat) (dd dh dw : pos)
  (bc depth h w : nat) : prop =
  SZ.fits (dw * (kw - 1) + 1) /\
  SZ.fits (w + 2 * pw) /\
  dw * (kw - 1) + 1 <= w + 2 * pw /\
  SZ.fits (pool_out_len_1d w kw sw pw dw * sw + kw * dw) /\
  SZ.fits (bc * depth * h * w) /\
  SZ.fits (bc * depth * h * pool_out_len_1d w kw sw pw dw) /\
  bc * depth * h * pool_out_len_1d w kw sw pw dw <=
    max_blocks * max_threads /\
  SZ.fits (dh * (kh - 1) + 1) /\
  SZ.fits (h + 2 * ph) /\
  dh * (kh - 1) + 1 <= h + 2 * ph /\
  SZ.fits (pool_out_len_1d h kh sh ph dh * sh + kh * dh) /\
  SZ.fits (bc * depth * pool_out_len_1d w kw sw pw dw * h) /\
  SZ.fits (bc * depth * pool_out_len_1d w kw sw pw dw *
    pool_out_len_1d h kh sh ph dh) /\
  bc * depth * pool_out_len_1d w kw sw pw dw *
    pool_out_len_1d h kh sh ph dh <= max_blocks * max_threads /\
  SZ.fits (dd * (kd - 1) + 1) /\
  SZ.fits (depth + 2 * pd) /\
  dd * (kd - 1) + 1 <= depth + 2 * pd /\
  SZ.fits (pool_out_len_1d depth kd sd pd dd * sd + kd * dd) /\
  SZ.fits (bc * (pool_out_len_1d h kh sh ph dh *
    pool_out_len_1d w kw sw pw dw) * depth) /\
  SZ.fits (bc * (pool_out_len_1d h kh sh ph dh *
    pool_out_len_1d w kw sw pw dw) * pool_out_len_1d depth kd sd pd dd) /\
  bc * (pool_out_len_1d h kh sh ph dh * pool_out_len_1d w kw sw pw dw) *
    pool_out_len_1d depth kd sd pd dd <= max_blocks * max_threads

unfold
let maxpool3d_raw_pre
  (k s p d b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) })
  (depth h w : szp) : prop =
  maxpool3d_full_pre k k k s s s p p p d d d (SZ.v (b *^ c)) depth h w

noeq type maxpool3d_full_result
  (kd kh kw sd sh sw : szp) (pd ph pw : sz) (dd dh dw : szp)
  (bc depth h w : szp) = {
  w_out : wo:szp {
    SZ.v wo == pool_out_len_1d w kw sw pw dw };
  h_out : ho:szp {
    SZ.v ho == pool_out_len_1d h kh sh ph dh };
  d_out : do_:szp {
    SZ.v do_ == pool_out_len_1d depth kd sd pd dd };
  output : array2 f32 (l2_bcm_pages bc (h_out * w_out) d_out);
}

unfold
let maxpool3d_raw_result
  (k s p d b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) })
  (depth h w : szp) : Type0 =
  maxpool3d_full_result k k k s s s p p p d d d (b *^ c) depth h w

unfold
let maxpool3d_full_post
  (kd kh kw sd sh sw : szp) (pd ph pw : sz) (dd dh dw : szp)
  (bc depth h w : szp)
  (sx : chest2 f32 (bc * depth * h) w)
  (r : maxpool3d_full_result kd kh kw sd sh sw pd ph pw dd dh dw
    bc depth h w) : slprop =
  on gpu_loc (r.output |->
    windowreduce_result reducer_fmax_f32
      (maxpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw
        r.w_out r.h_out sx)
      kd sd pd dd r.d_out)

unfold
let maxpool3d_raw_post
  (k s p d b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) })
  (depth h w : szp)
  (sx : chest2 f32 (b * c * depth * h) w)
  (r : maxpool3d_raw_result k s p d b c depth h w) : slprop =
  maxpool3d_full_post k k k s s s p p p d d d
    (b *^ c) depth h w sx r

(* KernelBench-shaped entry: derives [bc = b*c] and duplicates the scalar
   pooling parameters across D/H/W inside Kuiper. *)
fn maxpool3d_raw_alloc_f32
  (k s p d b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) })
  (depth h w : szp)
  (#sq_bd : squash (SZ.fits (SZ.v b * SZ.v c * SZ.v depth)))
  (#sq_bdh : squash (SZ.fits (SZ.v b * SZ.v c * SZ.v depth * SZ.v h)))
  (input : array2 f32 (l2_row_major (b * c * depth * h) w) { is_global input })
  (#fIn : perm)
  (#sx : chest2 f32 (b * c * depth * h) w)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (maxpool3d_full_pre k k k s s s p p p d d d
      (SZ.v (b *^ c)) depth h w)
  returns r : maxpool3d_full_result k k k s s s p p p d d d
    (b *^ c) depth h w
  ensures maxpool3d_full_post k k k s s s p p p d d d
    (b *^ c) depth h w sx r
