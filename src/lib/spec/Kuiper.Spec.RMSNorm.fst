module Kuiper.Spec.RMSNorm

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Approximates
open Kuiper.Approximates.Base
open Kuiper.Spec.Frobenius
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
    assert (Seq.equal (Seq.tail s) Seq.empty);
    Seq.lemma_eq_intro (Seq.tail s) Seq.empty;
    (* view_seq s reduces to SCons x Seq.empty; the fold then steps to
       [seq_fold_left f (f acc x) Seq.empty] which is [f acc x]. *)
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

(* Auxiliary sequence: pointwise [sq_step] applied to row [r] of [sx],
   prefix of length [k]. *)
let row_sq_prefix
  (#t:Type0) {| scalar t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot (Seq.seq t)
  = Seq.init_ghost k (fun j -> sq_step (acc2 sx r j))

(* The deterministic left-fold inside [reduce_batched] (with [sq_step]
   pre-map) equals [seq_fold_left add zero] over the row's squared
   prefix.  Inductive proof using the [row_reduce_partial_succ]
   SMTPat and [seq_fold_left_append]. *)
let rec row_reduce_partial_eq_fold
  (#t:Type0) {| scalar t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures
      row_reduce_partial (sq_step #t) sx r k ==
      seq_fold_left add zero (row_sq_prefix sx r k))
    (decreases k)
  = if k = 0 then begin
      assert (Seq.equal (row_sq_prefix sx r 0) Seq.empty);
      Seq.lemma_eq_intro (row_sq_prefix sx r 0) Seq.empty;
      ()
    end else begin
      row_reduce_partial_eq_fold sx r (k - 1);
      let last : t = sq_step (acc2 sx r (k - 1)) in
      let single : Seq.seq t = Seq.create 1 last in
      assert (Seq.equal (row_sq_prefix sx r k)
                        (Seq.append (row_sq_prefix sx r (k - 1)) single));
      Seq.lemma_eq_intro (row_sq_prefix sx r k)
                         (Seq.append (row_sq_prefix sx r (k - 1)) single);
      sfl_append add zero (row_sq_prefix sx r (k - 1)) single;
      seq_fold_left_one add
        (seq_fold_left add zero (row_sq_prefix sx r (k - 1)))
        last;
      ()
    end

(* Pointwise approximation of [sq_step]: [sq_step x %~ sq_step_r r]
   whenever [x %~ r].  Direct consequence of [a_mul]. *)
let sq_step_approx_pt
  (#t:Type0) {| scalar t, real_like t |}
  (x : t) (r : real)
  : Lemma (requires v_approximates x r)
          (ensures  v_approximates (sq_step x) (sq_step_r r))
  = a_mul x x r r

(* Indexwise approximation of the squared-prefix sequence and the
   real-valued [seq_map sq_step_r] of the row. *)
let row_sq_prefix_approx
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : natlt rows)
  : Lemma (row_sq_prefix sx r cols
           %~ KS.seq_map sq_step_r (to_real_seq (EM.ematrix_row sx r)))
  = let row_g : Seq.lseq t cols = EM.ematrix_row sx r in
    let row_r : Seq.seq real    = to_real_seq #t row_g in
    let lhs : Seq.seq t    = row_sq_prefix sx r cols in
    let rhs : Seq.seq real = KS.seq_map sq_step_r row_r in
    to_real_seq_is_approx #t row_g;
    assert (Seq.length lhs == cols);
    assert (Seq.length rhs == cols);
    let aux (j : nat{j < cols}) : Lemma ((lhs @! j) %~ (rhs @! j))
      = let xj : t    = acc2 sx r j in
        let rj : real = row_r @! j in
        assert (xj %~ rj);
        sq_step_approx_pt xj rj;
        ()
    in
    Classical.forall_intro aux;
    ()

(* The bridge lemma promised in the .fsti. *)
let row_reduce_partial_sq_approx
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : natlt rows)
  : Lemma (row_reduce_partial (sq_step #t) sx r cols
           %~ frobenius_sumsq_r (to_real_seq (EM.ematrix_row sx r)))
  = row_reduce_partial_eq_fold sx r cols;
    row_sq_prefix_approx sx r;
    sum_is_approx (row_sq_prefix sx r cols)
                  (KS.seq_map sq_step_r (to_real_seq (EM.ematrix_row sx r)));
    (* RHS unfolds to [rsum (seq_map sq_step_r ...)] by [frobenius_sumsq_r]
       and [rsum (s) == seq_fold_left (+.) 0R s] by definition. *)
    ()
