module Kuiper.KB.MinReduceDim

#lang-pulse
open Kuiper
open Kuiper.Math.Fmin
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.MinReduceDim
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module HRMin = Kuiper.Kernel.HReduce.Min
module Seq = FStar.Seq

(* Connect the kernel post [seq_reduce_rows_fmin] (per-row fmax fold,
   opaque) to [minreduce_post] (per-row [seq_fmin] of the row). *)
let bridge_post
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  (sy : chest1 f32 rows)
  : Lemma
      (requires sy == seq_to_chest1 (HRMin.seq_reduce_rows_fmin sx))
      (ensures  minreduce_post rows cols sx (chest1_to_seq sy))
  = let sys = HRMin.seq_reduce_rows_fmin sx in
    let aux (r : nat{r < rows})
      : Lemma (Seq.index (chest1_to_seq sy) r == seq_fmin (EM.ematrix_row sx r)) =
      assert (Seq.index (chest1_to_seq sy) r == acc1 sy r);
      assert (acc1 sy r == Seq.index sys r);
      assert (Seq.index sys r == HRMin.row_fmin sx r);
      HRMin.row_fmin_eq_seq_fmin sx r
    in
    Classical.forall_intro aux

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn minreduce_dim_fw_f32_impl
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads })
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
       pure (minreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy')))
{
  let bm : szp = b *^ m;
  assert pure (SZ.v bm == SZ.v b * SZ.v m);

  HRMin.reduce_batched_min_f32 bm d
    #_ #(c_l2_bcm_pages (SZ.v b) m d)
    #_ #(c_l1_forward _)
    x y;

  bridge_post #bm #d
    (reveal sx) (seq_to_chest1 (HRMin.seq_reduce_rows_fmin (reveal sx)));
  ()
}
#pop-options

fn minreduce_dim_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (y : array1 f32 (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  (#sy : chest1 f32 (SZ.v b * SZ.v m))
  preserves cpu ** on gpu_loc (x |-> sx)
  requires on gpu_loc (y |-> sy)
  ensures
    exists* (sy' : chest1 f32 (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy') **
      pure (minreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy'))
{
  minreduce_dim_fw_f32_impl b m d x y #sx #sy
}

fn minreduce_dim_alloc_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  norewrite
  preserves cpu ** on gpu_loc (x |-> sx)
  returns y : array1 f32 (l1_forward (SZ.v b * SZ.v m))
  ensures
    exists* (sy : chest1 f32 (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy) **
      pure (minreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy))
{
  let bm : szp = b *^ m;
  let y = alloc0 #f32 bm (l1_forward bm);
  minreduce_dim_fw_f32_impl b m d x y;
  y
}
