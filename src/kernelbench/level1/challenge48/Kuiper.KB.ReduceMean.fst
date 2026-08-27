module Kuiper.KB.ReduceMean

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.SumReduceDim
open Kuiper.Spec.MeanReduceDim
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map
module KS = Kuiper.Seq.Common
module HRedB = Kuiper.Kernel.HReduce.Block

(* Verified, extractable reciprocal 1/d as f32 (extracts to
   1.0f / (float)(int64_t)(uint64_t)d), so the mean divisor is computed
   inside the verification boundary instead of in unverified C. *)
let reducemean_recip_f32 (d : szp) : f32 =
  div one (of_int (FStar.Int.Cast.uint64_to_int64
                     (FStar.SizeT.sizet_to_uint64 d)))

(* Bridge lemma identical to ReduceSum.row_to_real_eq: a row of
   [to_real_matrix sx] equals [to_real_seq] of the corresponding row
   of [sx], as sequences. *)
let row_to_real_eq
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : nat { r < rows })
  : Lemma (Seq.equal
             (EM.ematrix_row (EM.to_real_matrix sx) r)
             (to_real_seq (EM.ematrix_row sx r)))
  = let lhs = EM.ematrix_row (EM.to_real_matrix sx) r in
    let rhs = to_real_seq (EM.ematrix_row sx r) in
    let aux (j:nat{j<cols}) : Lemma (Seq.index lhs j == Seq.index rhs j) = () in
    Classical.forall_intro aux

(* Per-row simplification: rsum (lseq_map id (row of to_real_matrix))
   = rsum (to_real_seq row).  Identical to ReduceSum.row_post_eq. *)
let row_post_eq
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : nat { r < rows })
  : Lemma
      (rsum (KS.lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r))
       == rsum (to_real_seq (EM.ematrix_row sx r)))
  = row_to_real_eq sx r;
    Seq.lemma_eq_intro
      (KS.lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r))
      (to_real_seq (EM.ematrix_row sx r))

(* For each row [r], if [smul_result inv_d s @! r == mul inv_d (s @! r)]
   and [s @! r %~ rsum (...)], then [meanreduce_row] holds with the
   witness [sumr = s @! r].  Helper lemma to discharge the per-row
   existential. *)
let mean_row_aux
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (inv_d : t)
  (sx : EM.chest2 t rows cols)
  (s_sum : chest1 t rows)
  (r : natlt rows)
  : Lemma
      (requires (acc1 s_sum r) %~ rsum (to_real_seq (EM.ematrix_row sx r)))
      (ensures
        row_mean #t inv_d sx
          (chest1_to_seq (chest_map (mul inv_d) s_sum)) r)
  = let s_after = chest1_to_seq (chest_map (mul inv_d) s_sum) in
    assert (Seq.index s_after r == mul inv_d (acc1 s_sum r))

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn reduce_mean_fw_f32_impl
  (b : szp)
  (m : szp { SZ.fits (SZ.v b * SZ.v m) /\ SZ.v b * SZ.v m <= max_blocks })
  (d : szp { SZ.v d <= max_threads /\
             SZ.fits (SZ.v d + max_threads) /\
             SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) })
  (inv_d : f32)
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (y : array1 f32 (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : EM.chest2 f32 (SZ.v b * SZ.v m) d)
  (#sy : chest1 f32 (SZ.v b * SZ.v m))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    on gpu_loc (y |-> sy)
  ensures
    on gpu_loc (x |-> sx) **
    (exists* (sy' : chest1 f32 (SZ.v b * SZ.v m)).
       on gpu_loc (y |-> sy') **
       pure (meanreduce_post (SZ.v b * SZ.v m) d inv_d sx (chest1_to_seq sy')))
{
  (* Build the real-valued ghost chest2 and the sx %~ vr witness. *)
  let bm : szp = b *^ m;
  assert pure (SZ.v bm == SZ.v b * SZ.v m);
  let vr : EM.chest2 real (SZ.v b * SZ.v m) d =
    hide (EM.to_real_matrix (reveal sx));
  assert pure (reveal sx %~ reveal vr);
  let vr' : EM.chest2 real bm d = vr;

  (* Launch 1: row-wise tree reduction (identity pre_map). *)
  HRedB.reduce_batched_block #f32 id id bm d 1024sz
    #_ #(c_l2_bcm_pages (SZ.v b) m d)
    #_ #(c_l1_forward _)
    x y vr';
  with sy1. assert (on gpu_loc (y |-> sy1));

  (* Bridge per-row block-reduce post into [sumreduce_post]. *)
  Classical.forall_intro
    (Classical.move_requires
       (row_post_eq #f32 (reveal sx)));
  assert pure (sumreduce_post (SZ.v b * SZ.v m) d (reveal sx) (chest1_to_seq (reveal sy1)));

  (* Launch 2: post-scale by inv_d in place. *)
  assert pure (SZ.v bm <= max_blocks * max_threads);
  Map.map_gpu (mul inv_d) bm #_ #(c_l1_forward _) y;

  (* Discharge per-row [meanreduce_post]. *)
  Classical.forall_intro
    (Classical.move_requires
       (mean_row_aux #f32 inv_d (reveal sx) (reveal sy1)));
  ()
}
#pop-options

let reduce_mean_fw_f32 : reduce_mean_fw_ty f32 = reduce_mean_fw_f32_impl
