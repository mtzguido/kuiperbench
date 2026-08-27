module Kuiper.KB.ArgminReduceDim

#lang-pulse
open Kuiper
open Kuiper.Math.Fmin
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.BCMPages
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module HRAmin = Kuiper.Kernel.HReduce.Argmin
module Seq = FStar.Seq
module I64 = FStar.Int64

(* Connect kernel post (per-row argmin_partial → i64) to the post: the
   first-occurrence bridge from row_argmin_first_at_full guarantees that
   the i64 index produced is in-bounds, points at the row-fmin, and is
   the first (smallest) column attaining it. *)
let bridge_post
  (#rows : nat) (cols : szp { SZ.v cols > 0 /\ SZ.v cols < pow2 63 })
  (sx : EM.chest2 f32 rows (SZ.v cols))
  (sy : chest1 I64.t rows)
  : Lemma
      (requires sy == seq_to_chest1 (HRAmin.seq_reduce_rows_argmin cols sx))
      (ensures  argminreduce_post rows (SZ.v cols) sx (chest1_to_seq sy))
  = let sys = HRAmin.seq_reduce_rows_argmin cols sx in
    let aux (r : nat{r < rows}) : Lemma
        (let bi = I64.v (Seq.index (chest1_to_seq sy) r) in
         0 <= bi /\ bi < SZ.v cols /\
         acc2 sx r bi == seq_fmin (EM.ematrix_row sx r) /\
         (forall (j : nat). j < bi ==>
            ~(acc2 sx r j == seq_fmin (EM.ematrix_row sx r)))) =
      let bi_nat = fst (HRAmin.row_argmin_partial sx r (SZ.v cols)) in
      HRAmin.row_argmin_idx_inv sx r (SZ.v cols);
      assert (Seq.index (chest1_to_seq sy) r == acc1 sy r);
      assert (acc1 sy r == Seq.index sys r);
      assert (Seq.index sys r == HRAmin.argmin_i64 cols sx r);
      assert (I64.v (Seq.index (chest1_to_seq sy) r) == bi_nat);
      HRAmin.row_argmin_first_at_full sx r
    in
    Classical.forall_intro aux

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn argminreduce_dim_fw_f32_impl
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads /\
             SZ.v d < pow2 63 })
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v m) (SZ.v d)) { is_global x })
  (y : array1 I64.t (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : erased (EM.chest2 f32 (SZ.v b * SZ.v m) (SZ.v d)))
  (#sy : erased (chest1 I64.t (SZ.v b * SZ.v m)))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    on gpu_loc (y |-> sy)
  ensures
    on gpu_loc (x |-> sx) **
    (exists* (sy' : chest1 I64.t (SZ.v b * SZ.v m)).
       on gpu_loc (y |-> sy') **
       pure (argminreduce_post (SZ.v b * SZ.v m) (SZ.v d) sx (chest1_to_seq sy')))
{
  let bm : szp = b *^ m;
  assert pure (SZ.v bm == SZ.v b * SZ.v m);

  HRAmin.reduce_batched_argmin_f32 bm d
    #_ #(c_l2_bcm_pages (SZ.v b) m d)
    #_ #(c_l1_forward _)
    x y;

  bridge_post #(SZ.v bm) d (reveal sx) (seq_to_chest1 (HRAmin.seq_reduce_rows_argmin d (reveal sx)));
  ()
}
#pop-options

fn argminreduce_dim_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads /\
             SZ.v d < pow2 63 })
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v m) (SZ.v d)) { is_global x })
  (y : array1 I64.t (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : erased (EM.chest2 f32 (SZ.v b * SZ.v m) (SZ.v d)))
  (#sy : erased (chest1 I64.t (SZ.v b * SZ.v m)))
  preserves cpu ** on gpu_loc (x |-> sx)
  requires on gpu_loc (y |-> sy)
  ensures
    exists* (sy' : chest1 I64.t (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy') **
      pure (argminreduce_post (SZ.v b * SZ.v m) (SZ.v d) sx (chest1_to_seq sy'))
{
  argminreduce_dim_fw_f32_impl b m d x y #sx #sy
}
