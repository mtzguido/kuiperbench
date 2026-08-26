module Kuiper.Kernel.RowLogSoftmax

#lang-pulse
open Kuiper
open Kuiper.Real { exp, log }
open Kuiper.EMatrix
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT
module KB = Kuiper.Kernel.HReduce.Block
module RB = Kuiper.Kernel.RowBroadcast
module LS = Kuiper.Kernel.LogSoftmax
module SM = Kuiper.Spec.Softmax
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Chest = Kuiper.Chest

(* Bridge the chest1 row-sum-of-exp to the seq form used by
   [reduce_batched_block]'s post-condition. *)
let row_sum_bridge (#m #n : nat) (ra1 : chest2 real m n) (i : natlt m)
  : Lemma (chest1_rsum (chest_map exp (chest2_row ra1 i))
           == rsum (lseq_map exp (ematrix_row ra1 i)))
  = Seq.lemma_eq_elim (chest1_to_seq (chest_map exp (chest2_row ra1 i)))
                      (lseq_map exp (ematrix_row ra1 i))

(* A cell of a chest2 row equals the matrix cell (through the [chest_slice]
   bijection; needs fuel to compute through it). *)
#push-options "--fuel 6 --ifuel 4"
let acc1_chest2_row (#et:Type) (#r #cc:nat) (x:chest2 et r cc) (i:natlt r) (j:natlt cc)
  : Lemma (acc1 (chest2_row x i) j == acc2 x i j)
  = ()
#pop-options

#push-options "--fuel 4 --ifuel 2"
let mk2_cell (#et : Type) (#d0 #d1 : nat)
  (f : natlt d0 -> natlt d1 -> GTot et) (i : natlt d0) (j : natlt d1)
  : Lemma (acc2 (mk2 f) i j == f i j)
  = ()
#pop-options

(* [acc1 (softmax_real row) j] in closed form (row-sum denominator nonzero). *)
#push-options "--fuel 6 --ifuel 2"
let softmax_real_cell (#n : nat { n > 0 }) (row : Chest.chest1 real n) (j : natlt n)
  : Lemma
      (requires chest1_rsum (chest_map exp row) =!= 0.0R)
      (ensures acc1 (SM.softmax_real row) j
               == exp (acc1 row j) /. chest1_rsum (chest_map exp row))
  = ()

(* [log_softmax_real = log . refine(>0) . softmax_real]; unfold one cell into
   the computed form [x - log (sum (exp x))].  The reals [log_div]/[log_exp]/
   [exp_positive] SMTPats close the last few steps. *)
let log_softmax_real_cell (#n : nat { n > 0 }) (row : Chest.chest1 real n) (j : natlt n)
  : Lemma
      (requires chest1_rsum (chest_map exp row) >. 0.0R)
      (ensures acc1 (LS.log_softmax_real row) j
               == acc1 row j -. log (chest1_rsum (chest_map exp row)))
  = softmax_real_cell row j
#pop-options

(* Fully-reduced cell of the golden spec, exposed via SMTPat (mirrors
   [Kuiper.Kernel.RowSoftmax.acc2_row_softmax_real]). *)
#push-options "--fuel 4 --ifuel 2"
let acc2_row_log_softmax_real (#m : nat) (#n : nat { n > 0 })
  (ra : chest2 real m n) (i : natlt m) (j : natlt n)
  : Lemma (acc2 (row_log_softmax_real #m #n ra) i j
           == acc2 ra i j -. log (chest1_rsum (chest_map exp (chest2_row ra i))))
          [SMTPat (acc2 (row_log_softmax_real #m #n ra) i j)]
  = // the row-sum-of-exp denominator is strictly positive
    sum_non_zero (lseq_map exp (ematrix_row ra i)) 0.0R;
    row_sum_bridge ra i;
    // row_log_softmax_real ra i j = acc1 (log_softmax_real (chest2_row ra i)) j
    mk2_cell (fun (i : natlt m) (j : natlt n) ->
                acc1 (LS.log_softmax_real (chest2_row ra i)) j) i j;
    // acc1 (log_softmax_real row) j = acc1 row j - log rsum
    log_softmax_real_cell (chest2_row ra i) j;
    // acc1 (chest2_row ra i) j = acc2 ra i j
    acc1_chest2_row ra i j
#pop-options

(* Glue: if every per-row sum approximates [rsum (exp row)], the log-subtract
   broadcast approximates [row_log_softmax_real].  [s_row_broadcast f a b]
   applies [f (broadcast i) (cell i j)], so the row sum is [f]'s FIRST argument
   and we compute [cell - log sum] from the SECOND. *)
#push-options "--z3rlimit 60"
let s_row_broadcast_approx_log_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#m : nat) (#n : nat { n > 0 })
  (sums : Chest.chest1 et m) (sa : chest2 et m n) (ra : chest2 real m n)
  : Lemma
      (requires
        sa %~ ra /\
        (forall (i : nat). i < m ==>
          v_approximates (acc1 sums i)
                         (rsum (lseq_map exp (ematrix_row ra i)))))
      (ensures RB.s_row_broadcast (fun (s:et) (aij:et) -> sub aij (flog s)) sums sa
               %~ row_log_softmax_real #m #n ra)
  = let lhs = RB.s_row_broadcast (fun (s:et) (aij:et) -> sub aij (flog s)) sums sa in
    let aux (idx : Kuiper.Shape.abs (m @| n @| INil))
      : Lemma (Chest.acc lhs idx %~ Chest.acc (row_log_softmax_real #m #n ra) idx) =
      let (i, (j, ())) = idx in
      let denom = chest1_rsum (chest_map exp (chest2_row ra i)) in
      // the row-sum denominator is strictly positive, hence log is well-defined
      sum_non_zero (lseq_map exp (ematrix_row ra i)) 0.0R;
      row_sum_bridge ra i;
      assert (denom == rsum (lseq_map exp (ematrix_row ra i)));
      assert (denom >. 0.0R);
      // sa %~ ra at this cell; sums approximates the denom
      assert (v_approximates (acc2 sa i j) (acc2 ra i j));
      assert (v_approximates (acc1 sums i) denom);
      // flog sums[i] %~ log denom, then cell - flog sums[i] %~ cell - log denom
      log_approx #et (acc1 sums i) denom;
      sub_approx #et (acc2 sa i j) (flog (acc1 sums i)) (acc2 ra i j) (log denom);
      // both sides reduce to canonical cell forms
      assert (Chest.acc lhs idx == sub (acc2 sa i j) (flog (acc1 sums i)));
      // [acc2_row_log_softmax_real] (SMTPat) reduces the golden-spec cell
      assert (acc2 (row_log_softmax_real #m #n ra) i j == acc2 ra i j -. log denom);
      ()
    in
    Classical.forall_intro aux
#pop-options

inline_for_extraction noextract
fn row_log_softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m : szp { m <= max_blocks })
  (n : szp { m * n <= max_blocks * max_threads })
  (#l : layout2 m n) {| ctlayout l |}
  (a : array2 et l { is_global a })
  (#sa : erased (chest2 et m n))
  (ra : erased (chest2 real m n))
  preserves cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  ensures
    exists* (sa' : chest2 et m n).
      on gpu_loc (a |-> sa') **
      pure (sa' %~ row_log_softmax_real ra)
{
  (* Step 1: one block per row, tree-reduce the row sums of exp(x). *)
  let sums = alloc0 #et m (l1_forward m);

  KB.reduce_batched_block #et fexp exp m n max_threads a sums ra;

  with sums_v. assert (on gpu_loc (sums |-> sums_v));

  (* Step 2: one thread per cell (i, j): a[i, j] := a_old[i, j] - log(sums[i]).
     [row_broadcast f] writes [f (broadcast i) (cell i j)], so the sum is the
     FIRST lambda argument and we subtract its log from the SECOND. *)
  RB.row_broadcast (fun (s:et) (aij:et) -> sub aij (flog s)) m n sums a;

  free sums;

  s_row_broadcast_approx_log_softmax #et #_ #_ #_ #(SZ.v m) #(SZ.v n) sums_v sa ra;
  ()
}
