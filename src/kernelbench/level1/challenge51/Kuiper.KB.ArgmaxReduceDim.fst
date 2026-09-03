module Kuiper.KB.ArgmaxReduceDim

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.BCMPages
module SZ = Kuiper.SizeT
module HRA = Kuiper.Kernel.HReduce.Argmax
module Seq = FStar.Seq
module I64 = FStar.Int64

inline_for_extraction noextract
instance sized_i64 : sized I64.t = { size = 8sz; default = 0L }

(* Connect the kernel's per-row result to the exact operational scan. *)
let bridge_post
  (#rows : nat) (cols : szp { SZ.v cols > 0 /\ SZ.v cols < pow2 63 })
  (sx : chest2 f32 rows cols)
  (sy : chest1 I64.t rows)
  : Lemma
      (requires sy == seq_to_chest1 (HRA.seq_reduce_rows_argmax cols sx))
      (ensures  argmaxreduce_post rows cols sx (chest1_to_seq sy))
  = let sys = HRA.seq_reduce_rows_argmax cols sx in
    let aux (r : nat{r < rows}) : Lemma
        (let bi = I64.v (Seq.index (chest1_to_seq sy) r) in
         0 <= bi /\ bi < SZ.v cols /\
         bi == fst (HRA.row_argmax_partial sx r cols)) =
      let bi_nat = fst (HRA.row_argmax_partial sx r cols) in
      HRA.row_argmax_idx_inv sx r cols;
      assert (Seq.index (chest1_to_seq sy) r == acc1 sy r);
      assert (acc1 sy r == Seq.index sys r);
      assert (Seq.index sys r == HRA.argmax_i64 cols sx r);
      assert (I64.v (Seq.index (chest1_to_seq sy) r) == bi_nat)
    in
    Classical.forall_intro aux

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn argmaxreduce_dim_fw_f32_impl
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads /\
             SZ.v d < pow2 63 })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (y : array1 I64.t (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  (#sy : chest1 I64.t (SZ.v b * SZ.v m))
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (y |-> sy)
  ensures
    (exists* (sy' : chest1 I64.t (SZ.v b * SZ.v m)).
       on gpu_loc (y |-> sy') **
       pure (argmaxreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy')))
{
  let bm : szp = b *^ m;
  assert pure (SZ.v bm == SZ.v b * SZ.v m);

  HRA.reduce_batched_argmax_f32 bm d
    #_ #(c_l2_bcm_pages (SZ.v b) m d)
    #_ #(c_l1_forward _)
    x y;

  bridge_post #bm d (reveal sx) (seq_to_chest1 (HRA.seq_reduce_rows_argmax d (reveal sx)));
  ()
}
#pop-options

fn argmaxreduce_dim_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads /\
             SZ.v d < pow2 63 })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (y : array1 I64.t (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  (#sy : chest1 I64.t (SZ.v b * SZ.v m))
  preserves cpu ** on gpu_loc (x |-> sx)
  requires on gpu_loc (y |-> sy)
  ensures
    exists* (sy' : chest1 I64.t (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy') **
      pure (argmaxreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy'))
{
  argmaxreduce_dim_fw_f32_impl b m d x y #sx #sy
}

fn argmaxreduce_dim_alloc_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads /\
             SZ.v d < pow2 63 })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  norewrite
  preserves cpu ** on gpu_loc (x |-> sx)
  returns y : array1 I64.t (l1_forward (SZ.v b * SZ.v m))
  ensures
    exists* (sy : chest1 I64.t (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy) **
      pure (argmaxreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy))
{
  let bm : szp = b *^ m;
  let y = alloc0 #I64.t bm (l1_forward bm);
  argmaxreduce_dim_fw_f32_impl b m d x y;
  y
}
