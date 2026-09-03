module Kuiper.Spec.L1Norm

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Approximates
open Kuiper.Approximates.Base
open Kuiper.Floating.Base
open Kuiper.Kernel.HReduce
open Kuiper.Seq.Common
open Kuiper.Chest
module Seq = FStar.Seq
module EM = Kuiper.EMatrix
module KS = Kuiper.Seq.Common

(* seq_fold_left over a singleton reduces to one application of [f]. *)
let seq_fold_left_one (#a #b : Type) (f : b -> a -> b) (acc : b) (x : a)
  : Lemma (seq_fold_left f acc (Seq.create 1 x) == f acc x)
  = let s : Seq.seq a = Seq.create 1 x in
    assert (Seq.length s == 1);
    assert (Seq.head s == x);
    Seq.lemma_eq_intro (Seq.tail s) Seq.empty;
    ()

(* Local copy of [Kuiper.Seq.Common.seq_fold_left_append] -- not
   exported through the .fsti, so we re-prove it inline. *)
let rec sfl_append
  (#a #b : Type) (f : b -> a -> b) (acc : b) (l0 l1 : Seq.seq a)
  : Lemma (ensures seq_fold_left f acc (Seq.append l0 l1) ==
                   seq_fold_left f (seq_fold_left f acc l0) l1)
          (decreases Seq.length l0)
  = match view_seq l0 with
    | SNil -> assert (Seq.equal (Seq.append l0 l1) l1)
    | SCons hd tl ->
      let SCons hd' tl' = view_seq (Seq.append l0 l1) in
      assert (hd == hd');
      assert (Seq.equal tl' (Seq.append tl l1));
      sfl_append f (f acc hd) tl l1

(* Auxiliary sequence: pointwise [l1_abs] applied to row [r] of [sx],
   prefix of length [k]. *)
let row_l1_abs_prefix
  (#t:Type0) {| scalar t, floating t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot (Seq.seq t)
  = Seq.init_ghost k (fun j -> l1_abs (acc2 sx r j))

(* The deterministic left-fold inside [reduce_batched] (with [l1_abs]
   pre-map) equals [seq_fold_left add zero] over the row's abs-ed
   prefix.  Inductive proof using the [row_reduce_partial_succ]
   SMTPat and [seq_fold_left_append]. *)
let rec row_reduce_partial_eq_fold
  (#t:Type0) {| scalar t, floating t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures
      row_reduce_partial (l1_abs #t) sx r k ==
      seq_fold_left add zero (row_l1_abs_prefix sx r k))
    (decreases k)
  = if k = 0 then begin
      Seq.lemma_eq_intro (row_l1_abs_prefix sx r 0) Seq.empty;
      ()
    end else begin
      row_reduce_partial_eq_fold sx r (k - 1);
      let last : t = l1_abs (acc2 sx r (k - 1)) in
      let single : Seq.seq t = Seq.create 1 last in
      Seq.lemma_eq_intro (row_l1_abs_prefix sx r k)
                         (Seq.append (row_l1_abs_prefix sx r (k - 1)) single);
      sfl_append add zero (row_l1_abs_prefix sx r (k - 1)) single;
      seq_fold_left_one add
        (seq_fold_left add zero (row_l1_abs_prefix sx r (k - 1)))
        last;
      ()
    end

let l1_abs_approx
  (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (x : t) (rx : real)
  : Lemma (requires x %~ rx)
          (ensures l1_abs x %~ l1_abs_r rx)
  = assert (zero #t %~ 0.0R);
    sub_approx (zero #t) x 0.0R rx;
    fmax_approx x (sub zero x) rx (0.0R -. rx)

let row_l1_abs_prefix_approx
  (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (rx : chest2 real rows cols { sx %~ rx })
  (r : natlt rows)
  : Lemma (row_l1_abs_prefix sx r cols
           %~ KS.lseq_map l1_abs_r (EM.ematrix_row rx r))
  = let lhs : Seq.seq t = row_l1_abs_prefix sx r cols in
    let rhs : Seq.seq real = KS.lseq_map l1_abs_r (EM.ematrix_row rx r) in
    let aux (j : nat { j < cols }) : Lemma ((lhs @! j) %~ (rhs @! j))
      = assert (acc2 sx r j %~ acc2 rx r j);
        l1_abs_approx (acc2 sx r j) (acc2 rx r j)
    in
    Classical.forall_intro aux

(* The bridge lemma promised in the .fsti. *)
let row_reduce_partial_l1_abs_approx
  (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (rx : chest2 real rows cols { sx %~ rx })
  (r : natlt rows)
  : Lemma (row_reduce_partial (l1_abs #t) sx r cols
           %~ l1_sum_r (EM.ematrix_row rx r))
  = let abs_r : Seq.lseq real cols =
      KS.lseq_map l1_abs_r (EM.ematrix_row rx r) in
    row_reduce_partial_eq_fold sx r cols;
    row_l1_abs_prefix_approx sx rx r;
    sum_is_approx (row_l1_abs_prefix sx r cols) abs_r
