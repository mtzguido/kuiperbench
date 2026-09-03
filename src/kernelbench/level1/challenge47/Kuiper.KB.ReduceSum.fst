module Kuiper.KB.ReduceSum

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.SumReduceDim
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module KB = Kuiper.Kernel.HReduce.Block
module KS = Kuiper.Seq.Common

(* Bridge lemma: a row of [EM.to_real_matrix sx] equals [to_real_seq] of
   the corresponding row of [sx], as sequences.  Both sides are
   length-[cols] sequences whose [j]-th element is [to_real (acc2 sx r j)]. *)
let row_to_real_eq
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (r : nat { r < rows })
  : Lemma (Seq.equal
             (EM.ematrix_row (EM.to_real_matrix sx) r)
             (to_real_seq (EM.ematrix_row sx r)))
  = let lhs = EM.ematrix_row (EM.to_real_matrix sx) r in
    let rhs = to_real_seq (EM.ematrix_row sx r) in
    let aux (j:nat{j<cols}) : Lemma (Seq.index lhs j == Seq.index rhs j) = () in
    Classical.forall_intro aux

(* Per-row simplification of the [reduce_batched_block] postcondition
   into the form used by [sumreduce_post]:
       rsum (lseq_map id (ematrix_row (to_real_matrix sx) r))
     = rsum (to_real_seq (ematrix_row sx r)). *)
let row_post_eq
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (r : nat { r < rows })
  : Lemma
      (rsum (KS.lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r))
       == rsum (to_real_seq (EM.ematrix_row sx r)))
  = row_to_real_eq sx r;
    Seq.lemma_eq_intro
      (KS.lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r))
      (to_real_seq (EM.ematrix_row sx r))

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn reduce_sum_fw_f32_impl
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks /\
             SZ.fits (SZ.v d + max_threads) })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (y : array1 f32 (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  (#sy : chest1 f32 (SZ.v b * SZ.v m))
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (y |-> sy)
  ensures
    (exists* (sy' : chest1 f32 (SZ.v b * SZ.v m)).
       on gpu_loc (y |-> sy') **
       pure (sumreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy')))
{
  let bm : szp = b *^ m;
  assert pure (SZ.v bm == SZ.v b * SZ.v m);
  (* Build the real-valued ghost chest2 and the sx %~ vr witness. *)
  let vr : chest2 real (SZ.v b * SZ.v m) d =
    hide (EM.to_real_matrix (reveal sx));
  assert pure (reveal sx %~ reveal vr);
  let vr' : chest2 real bm d = vr;
  KB.reduce_batched_block #f32 id id bm d 1024sz
    #_ #(c_l2_bcm_pages (SZ.v b) m d)
    #_ #(c_l1_forward _)
    x y vr';
  with sy'. assert (on gpu_loc (y |-> sy'));
  (* Bridge per-row post into [sumreduce_post]. *)
  Classical.forall_intro
    (Classical.move_requires
       (row_post_eq #f32 #_ #_ #(SZ.v b * SZ.v m) #d (reveal sx)));
  ()
}
#pop-options

let reduce_sum_fw_f32 = reduce_sum_fw_f32_impl

fn reduce_sum_alloc_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks /\
             SZ.fits (SZ.v d + max_threads) })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  norewrite
  preserves cpu ** on gpu_loc (x |-> sx)
  returns y : array1 f32 (l1_forward (SZ.v b * SZ.v m))
  ensures
    exists* (sy : chest1 f32 (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy) **
      pure (sumreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy))
{
  let bm : szp = b *^ m;
  let y = alloc0 #f32 bm (l1_forward bm);
  reduce_sum_fw_f32_impl b m d x y;
  y
}
