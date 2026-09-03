module Kuiper.KB.MaxPool3D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout { is_full }
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { reducer_fmax_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce, windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
open Kuiper.Tensor.Layout.BCMPages { l2_bcm_pages, c_l2_bcm_pages }
open Kuiper.Array2.Recast { recast_gpu }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module ML = FStar.Math.Lemmas

(* Verified computation of the PyTorch 1-D pool output length, used inside
   the complete verified entry and provably equal to [pool_out_len_1d].
   [s], [k], [d] are szp (>= 1) so the spec's [s = 0] branch is dead and
   the SZ subtraction [padded -^ kspan] is guarded by the [padded <^ kspan]
   test exactly as in the spec. *)
let pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)
  =
  let kspan  : sz = SZ.((d *^ (k -^ 1sz)) +^ 1sz) in
  let padded : sz = SZ.(l +^ (2sz *^ p)) in
  if SZ.(padded <^ kspan) then 0sz
  else SZ.(((padded -^ kspan) /^ s) +^ 1sz)

inline_for_extraction noextract
fn maxpool3d_axis_fw
  (#t : Type0) {| scalar t |}
  (m_inst : Kuiper.Monoid.Reduce.reducer t)
  (k s p d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l    : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
  (#lin  : layout2 bc l)     {| ctlayout lin  |}
  (#lout : layout2 bc l_out) {| ctlayout lout |}
  (input  : array2 t lin  { is_global input  })
  (output : array2 t lout { is_global output })
  (#fIn  : perm)
  (#sx   : chest2 t bc l)
  (#sout : chest2 t bc l_out)
  preserves
    cpu **
    on gpu_loc (input  |-> Frac fIn sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
  ensures
    on gpu_loc (output |->
      windowreduce_result m_inst sx
        k s p d l_out)
{
  windowreduce m_inst k s p d bc l l_out input output
}

inline_for_extraction noextract
let maxpool3d_axis_fw_f32 =
  fun k s p d bc l l_out #_ #_ #_ #_ input output #fIn #sx #sout ->
    maxpool3d_axis_fw #f32 reducer_fmax_f32 k s p d bc l l_out input output
      #fIn #sx #sout

let maxpool3d_axis_fw_rm_f32 =
  fun k s p d bc l l_out input output #fIn #sx #sout ->
    maxpool3d_axis_fw_f32 k s p d bc l l_out
      #(l2_row_major bc l)     #_
      #(l2_row_major bc l_out) #_
      input output
      #fIn #sx #sout

(* Upper bound on the pool output length: [L_out <= L + 2P].  [kspan >= 1] and
   [S >= 1] give [(padded - kspan)/S <= padded - kspan <= padded - 1 < padded].
   This lets the self-allocating axis entry below discharge ALL the
   [l_out]-dependent overflow / launch-bound side conditions from bounds stated
   purely on the raw axis dims [(bc, l)], so the bridge never computes
   [l_out]. *)
let pool_out_len_1d_ub (l k s p d : nat)
  : Lemma (requires k >= 1 /\ s >= 1 /\ d >= 1)
          (ensures pool_out_len_1d l k s p d <= l + 2 * p)
  = let kspan = d * (k - 1) + 1 in
    let padded = l + 2 * p in
    if padded < kspan || s = 0 then ()
    else begin
      ML.lemma_div_mod (padded - kspan) s;
      assert ((padded - kspan) / s <= padded - kspan)
    end

(* Internal self-allocating per-axis pass used by the complete composition. *)
inline_for_extraction noextract
fn maxpool3d_axis_alloc
  (k s p d : szp)
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
    pure (SZ.fits ((SZ.v l + 2 * SZ.v p) * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.fits (SZ.v bc * (SZ.v l + 2 * SZ.v p))) **
    pure (SZ.v bc * (SZ.v l + 2 * SZ.v p) <= max_blocks * max_threads)
  returns r : (lo:sz { SZ.v lo == pool_out_len_1d l k s p d }
               & array2 f32 (l2_row_major bc lo))
  ensures
    on gpu_loc ((dsnd r) |->
      windowreduce_result reducer_fmax_f32 sx
        k s p d (dfst r)) **
    pure (SZ.v (dfst r) ==
            pool_out_len_1d l k s p d)
{
  let l_out = pool_out_len_1d_sz l k s p d;
  pool_out_len_1d_ub l k s p d;
  ML.lemma_mult_le_left bc l_out (SZ.v l + 2 * SZ.v p);
  ML.lemma_mult_le_right s l_out (SZ.v l + 2 * SZ.v p);
  let output = alloc0 #f32 (bc *^ l_out) (l2_row_major bc l_out);
  maxpool3d_axis_fw_rm_f32 k s p d bc l l_out input output;
  (| (l_out <: (lo:sz { SZ.v lo == pool_out_len_1d l k s p d })), output |)
}

let maxpool3d_axis_alloc_f32 =
  fun k s p d bc l input #fIn #sx ->
    maxpool3d_axis_alloc k s p d bc l input #fIn #sx

(* A fitting window and positive stride produce a nonempty output axis. *)
let pool_out_len_1d_pos (l k s p d : nat)
  : Lemma (requires k >= 1 /\ s >= 1 /\ d >= 1 /\ d * (k - 1) + 1 <= l + 2 * p)
          (ensures pool_out_len_1d l k s p d >= 1)
  = ()

(* Reassociation/commutation facts for the two zero-copy recasts. *)
let prod3_comm (a x y : nat) : Lemma (a * x * y == a * y * x) = ()
let prod4_rotate (a b c d : nat)
  : Lemma (a * b * c * d == a * (d * c) * b) = ()

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn maxpool3d_full_alloc_f32
  (kd kh kw sd sh sw pd ph pw dd dh dw : szp)
  (bc depth h w : szp)
  (#_ : squash (SZ.fits (SZ.v bc * SZ.v depth)))
  (#_ : squash (SZ.fits (SZ.v bc * SZ.v depth * SZ.v h)))
  (input : array2 f32 (l2_row_major (bc * depth * h) w) { is_global input })
  (#fIn : perm)
  (#sx : chest2 f32 (bc * depth * h) w)
  preserves
    cpu **
    on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (maxpool3d_full_pre kd kh kw sd sh sw pd ph pw dd dh dw
      bc depth h w)
  returns r : maxpool3d_full_result kd kh kw sd sh sw pd ph pw dd dh dw
    bc depth h w
  ensures maxpool3d_full_post kd kh kw sd sh sw pd ph pw dd dh dw
    bc depth h w sx r
{
  (* Pass 1: reduce contiguous W rows. *)
  let wo0 = pool_out_len_1d_sz w kw sw pw dw;
  pool_out_len_1d_pos w kw sw pw dw;
  let wo : (x:sz { SZ.v x == pool_out_len_1d w kw sw pw dw /\ SZ.v x > 0 }) = wo0;
  let rows_w : szp = (bc *^ depth) *^ h;
  let mid_w = alloc0 #f32 (rows_w *^ wo) (l2_row_major rows_w wo);
  maxpool3d_axis_fw_rm_f32 kw sw pw dw rows_w w wo input mid_w;

  (* Pass 2: recast row-major (bc,D,H,Wout) as rows (bc*D*Wout,H). *)
  prod3_comm (SZ.v bc * SZ.v depth) (SZ.v h) (SZ.v wo);
  let mid_h_in = recast_gpu
    (l2_bcm_pages (SZ.v bc * SZ.v depth) (SZ.v wo) (SZ.v h)) mid_w;
  let ho0 = pool_out_len_1d_sz h kh sh ph dh;
  pool_out_len_1d_pos h kh sh ph dh;
  let ho : (x:sz { SZ.v x == pool_out_len_1d h kh sh ph dh /\ SZ.v x > 0 }) = ho0;
  let rows_h : szp = (bc *^ depth) *^ wo;
  let mid_h = alloc0 #f32 (rows_h *^ ho)
    (l2_bcm_pages (SZ.v bc * SZ.v depth) (SZ.v wo) (SZ.v ho));
  maxpool3d_axis_fw #f32 reducer_fmax_f32 kh sh ph dh rows_h h ho
    #(l2_bcm_pages (SZ.v bc * SZ.v depth) (SZ.v wo) (SZ.v h))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc * SZ.v depth)) wo h)
    #(l2_bcm_pages (SZ.v bc * SZ.v depth) (SZ.v wo) (SZ.v ho))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc * SZ.v depth)) wo ho)
    mid_h_in mid_h;
  free mid_h_in;

  (* Pass 3: recast (bc,D,Hout,Wout) as rows (bc*Hout*Wout,D). *)
  prod4_rotate (SZ.v bc) (SZ.v depth) (SZ.v wo) (SZ.v ho);
  let mid_d_in = recast_gpu
    (l2_bcm_pages (SZ.v bc) (SZ.v ho * SZ.v wo) (SZ.v depth)) mid_h;
  let do0 = pool_out_len_1d_sz depth kd sd pd dd;
  pool_out_len_1d_pos depth kd sd pd dd;
  let do_ : (x:sz { SZ.v x == pool_out_len_1d depth kd sd pd dd /\ SZ.v x > 0 }) = do0;
  let how : szp = ho *^ wo;
  let rows_d : szp = bc *^ how;
  let out = alloc0 #f32 (rows_d *^ do_)
    (l2_bcm_pages (SZ.v bc) (SZ.v how) (SZ.v do_));
  maxpool3d_axis_fw #f32 reducer_fmax_f32 kd sd pd dd rows_d depth do_
    #(l2_bcm_pages (SZ.v bc) (SZ.v how) (SZ.v depth))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc)) how depth)
    #(l2_bcm_pages (SZ.v bc) (SZ.v how) (SZ.v do_))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc)) how do_)
    mid_d_in out;
  free mid_d_in;
  (| wo, (| ho, (| do_, out |) |) |)
}
#pop-options

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
    (SZ.v (b *^ c)) depth h w
  ensures maxpool3d_full_post k k k s s s p p p d d d
    (SZ.v (b *^ c)) depth h w sx r
{
  maxpool3d_full_alloc_f32 k k k s s s p p p d d d
    (b *^ c) depth h w #sq_bd #sq_bdh input #fIn #sx
}
