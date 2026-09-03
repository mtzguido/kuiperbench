module Kuiper.KB.AvgPool3D

(* KernelBench L1 #46: AvgPool3D.
 *
 * 3-D average pooling (with PyTorch's default count_include_pad=True
 * and stride defaulting to kernel_size) reduces to three passes of the
 * verified [Kuiper.Kernel.WindowReduce1D] primitive instantiated with
 * [reducer_fadd_f32] (rid = 0.0f, rop = +) plus a verified physical-order
 * scale by the f32 reciprocal of that axis's kernel extent.
 *
 *   pass 1: per-row sum over W (B*C*D*H rows of length W); then scale /kW
 *   pass 2: per-row sum over H (B*C*D*W_out rows of length H); then scale /kH
 *   pass 3: per-row sum over D (B*C*W_out*H_out rows of length D); then scale /kD
 *
 * The public [avgpool3d_raw_alloc_f32] entry owns this full sequence,
 * including allocation, both zero-copy layout recasts, and all three scales.
 * Its post states the exact implementation-order f32 composition; it assumes
 * no real or floating-point associativity.  The bridge checks refinements and
 * invokes this entry once.
 *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout { from_seq, to_seq, is_full }
open Kuiper.Tensor.Layout.Alg { l2_row_major, l1_forward }
open Kuiper.Monoid.Reduce.F32 { reducer_fadd_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
open Kuiper.Tensor.Layout.BCMPages { l2_bcm_pages }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix

(* Verified, extractable 1-D pool output-length formula (see .fst), provably
   equal to the pure spec [pool_out_len_1d].  [p] is [sz] (>= 0). *)
val pool_out_len_1d_sz
  (l k s : szp) (p : sz) (d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)

(* Verified, extractable reciprocal 1/k as f32 (see .fst). *)
val avgpool_recip_f32 (k : szp)
  : r:f32 { r %~ (1.0R /. FStar.Real.of_int (SZ.v k)) }

(* Verification-facing wrapper type (layout-polymorphic, f32 carrier). *)
inline_for_extraction noextract
fn avgpool3d_axis_fw_f32
  (k s : szp)
(p : sz)
(d : szp)
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
   windowreduce_result reducer_fadd_f32 sx
     k s p d l_out)


(* Concrete-layout extractable entry (l2_row_major). *)
fn avgpool3d_axis_fw_rm_f32
  (k s : szp)
(p : sz)
(d : szp)
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
   windowreduce_result reducer_fadd_f32 sx
     k s p d l_out)


(* Internal self-allocating per-axis building block.  Takes the raw per-axis
   dims [(k, s, p, d)], the leading product [bc] (the non-reduced dims folded
   into rows for this pass), the reduced axis length [l], and the input tensor.
   It computes [l_out] via the verified [pool_out_len_1d_sz], allocates the
   [(bc, l_out)] GPU output buffer, fills it with the per-window SUM
   (reducer_fadd_f32, rid = 0, padding -> 0), divides every element by [K] in
   place via the verified [Kuiper.KB.ScalarMul] kernel (scaling by
   [inv_k = avgpool_recip_f32 k = 1/K]), and returns the pair
   [(l_out, output_buffer)] — ownership passes to the caller.  The post is
   exactly "windowed sum, then /K": every output accumulator equals [inv_k]
   times the corresponding [windowreduce_result] (sum) accumulator.  Applying
   this per-pass /K across the three (3-D) separable axis passes yields the
   PyTorch divisor K*K*K (count_include_pad = True).

   It is composed internally by the complete public entry; the bridge does not
   invoke individual axis passes. *)
fn avgpool3d_axis_alloc_f32
  (k s : szp)
(p : sz)
(d : szp)
(bc : szp { SZ.v bc <= max_blocks * max_threads })
(l : szp { SZ.fits (SZ.v bc * SZ.v l) })
(input : array2 f32 (l2_row_major bc l) { is_global input })
(#fIn : perm)
(#sx  : chest2 f32 bc l)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
 pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
 pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
 pure (SZ.fits (pool_out_len_1d l k s p d
                  * SZ.v s + SZ.v k * SZ.v d)) **
 pure (SZ.fits (SZ.v bc *
         pool_out_len_1d l k s p d)) **
 pure (SZ.v bc *
         pool_out_len_1d l k s p d
       <= max_blocks * max_threads)
returns r : (lo:sz { SZ.v lo == pool_out_len_1d l k s p d }
            & array2 f32 (l2_row_major bc lo))
ensures
 on gpu_loc ((dsnd r) |->
   mk2 (fun (i:natlt bc) (j:natlt (dfst r)) ->
     mul (avgpool_recip_f32 k)
         (acc2 (windowreduce_result reducer_fadd_f32 sx
                    k s p d (dfst r)) i j))) **
 pure (SZ.v (dfst r) ==
         pool_out_len_1d l k s p d)

unfold
let avgpool3d_axis_result
  (k : szp) (#rows : nat) (#l : nat) (sx : chest2 f32 rows l)
  (s p d : nat) (l_out : nat)
  : chest2 f32 rows l_out
  = mk2 (fun (i:natlt rows) (j:natlt l_out) ->
      mul (avgpool_recip_f32 k)
        (acc2 (windowreduce_result reducer_fadd_f32 sx k s p d l_out) i j))

(* Exact result of scaling every physical slot of an arbitrary full layout.
   This deliberately states implementation order without assuming any
   floating-point algebra. *)
unfold
let avgpool3d_scale_layout_result
  (#rows : nat) (#cols : nat)
  (layout : layout2 rows cols { is_full layout })
  (c : f32) (sx : chest2 f32 rows cols)
  : chest2 f32 rows cols
  = let n = rows * cols in
    let flat = l1_forward n in
    from_seq layout
      (to_seq flat
        (chest_map (mul c) (from_seq flat (to_seq layout sx))))

[@@"opaque_to_smt"]
let avgpool3d_axis_layout_result
  (#rows : nat) (#l : nat) (#l_out : nat)
  (layout : layout2 rows l_out { is_full layout })
  (k : szp) (sx : chest2 f32 rows l)
  (s p d : nat)
  : chest2 f32 rows l_out
  = avgpool3d_scale_layout_result layout (avgpool_recip_f32 k)
      (windowreduce_result reducer_fadd_f32 sx k s p d l_out)

[@@"opaque_to_smt"]
let avgpool3d_mid_w_view
  (bc depth h w : nat)
  (kw : szp) (sw : pos) (pw : nat) (dw : pos)
  (w_out : pos)
  (sx : chest2 f32 (bc * depth * h) w)
  : chest2 f32 (bc * depth * w_out) h
  = from_seq (l2_bcm_pages (bc * depth) w_out h)
      (to_seq (l2_row_major (bc * depth * h) w_out)
        (avgpool3d_axis_layout_result
          (l2_row_major (bc * depth * h) w_out) kw sx sw pw dw))

[@@"opaque_to_smt"]
let avgpool3d_mid_h_view
  (bc depth h w : nat)
  (kh kw : szp) (sh sw : pos) (ph pw : nat) (dh dw : pos)
  (w_out h_out : pos)
  (sx : chest2 f32 (bc * depth * h) w)
  : chest2 f32 (bc * (h_out * w_out)) depth
  = from_seq (l2_bcm_pages bc (h_out * w_out) depth)
      (to_seq (l2_bcm_pages (bc * depth) w_out h_out)
        (avgpool3d_axis_layout_result
          (l2_bcm_pages (bc * depth) w_out h_out) kh
          (avgpool3d_mid_w_view bc depth h w kw sw pw dw w_out sx)
          sh ph dh))

(* Host validation establishes these arithmetic side conditions as one pure
   fact.  Naming the complete boundary keeps Pulse from repeatedly elaborating
   the large dependent layout context for each individual conjunct. *)
unfold
let avgpool3d_full_pre
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
  bc * (pool_out_len_1d h kh sh ph dh *
    pool_out_len_1d w kw sw pw dw) * pool_out_len_1d depth kd sd pd dd <=
    max_blocks * max_threads

unfold
let avgpool3d_full_result
  (kd kh kw sd sh sw : pos) (pd ph pw : nat) (dd dh dw : pos)
  (bc depth h w : nat) : Type0 =
  (wo : sz { SZ.v wo == pool_out_len_1d w kw sw pw dw /\ SZ.v wo > 0 }
   & (ho : sz { SZ.v ho == pool_out_len_1d h kh sh ph dh /\ SZ.v ho > 0 }
      & (do_ : sz { SZ.v do_ == pool_out_len_1d depth kd sd pd dd /\
                    SZ.v do_ > 0 }
         & array2 f32 (l2_bcm_pages bc (ho * wo) do_))))

unfold
let avgpool3d_full_post
  (kd kh kw : szp) (sd sh sw : pos) (pd ph pw : nat) (dd dh dw : pos)
  (bc depth h w : nat)
  (sx : chest2 f32 (bc * depth * h) w)
  (r : avgpool3d_full_result kd kh kw sd sh sw pd ph pw dd dh dw
    bc depth h w) : slprop =
  on gpu_loc ((dsnd (dsnd (dsnd r))) |->
    avgpool3d_axis_layout_result
      (l2_bcm_pages bc (dfst (dsnd r) * dfst r) (dfst (dsnd (dsnd r)))) kd
      (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw
        (dfst r) (dfst (dsnd r)) sx)
      sd pd dd)

(* Complete count-include-pad AvgPool3D pipeline.  Each axis performs the
   implementation-order f32 sum followed by its verified reciprocal scale;
   recasts change only the verified layout, never the bytes. *)
fn avgpool3d_full_alloc_f32
  (kd kh kw sd sh sw : szp) (pd ph pw : sz) (dd dh dw : szp)
  (bc depth h w : szp)
  (#_ : squash (SZ.fits (SZ.v bc * SZ.v depth)))
  (#_ : squash (SZ.fits (SZ.v bc * SZ.v depth * SZ.v h)))
  (input : array2 f32 (l2_row_major (bc * depth * h) w) { is_global input })
  (#fIn : perm)
  (#sx : chest2 f32 (bc * depth * h) w)
preserves
  cpu ** on gpu_loc (input |-> Frac fIn sx)
requires
  pure (avgpool3d_full_pre kd kh kw sd sh sw pd ph pw dd dh dw
    bc depth h w)
returns r : avgpool3d_full_result kd kh kw sd sh sw pd ph pw dd dh dw
  bc depth h w
ensures avgpool3d_full_post kd kh kw sd sh sw pd ph pw dd dh dw
  bc depth h w sx r

(* KernelBench-facing entry: derives [bc = b*c], broadcasts the scalar pool
   parameters across the three axes, and supplies the implicit unit dilation. *)
fn avgpool3d_raw_alloc_f32
  (k s : szp) (p : sz) (b : szp)
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
    pure (avgpool3d_full_pre k k k s s s p p p 1 1 1
      (SZ.v (b *^ c)) depth h w)
  returns r : avgpool3d_full_result k k k s s s p p p 1 1 1
    (SZ.v (b *^ c)) depth h w
  ensures avgpool3d_full_post k k k s s s p p p 1 1 1
    (SZ.v (b *^ c)) depth h w sx r
