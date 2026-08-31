module Kuiper.Spec.Frobenius

open Kuiper.Real
open Kuiper.Seq.Common
open Kuiper.Seq.Common { (@!) }
module Seq = FStar.Seq

let rec rsum_nonnegative
  (s : Seq.seq real)
  : Lemma
      (requires forall (i : nat). i < Seq.length s ==> Seq.index s i >=. 0.0R)
      (ensures rsum s >=. 0.0R)
      (decreases Seq.length s)
  = match view_seq s with
    | SNil -> ()
    | SCons hd tl ->
      rsum_nonnegative tl;
      rsum_append (Seq.create 1 hd) tl

let frobenius_sumsq_nonnegative (s : Seq.seq real)
  : Lemma (frobenius_sumsq_r s >=. 0.0R)
  = let terms = seq_map sq_step_r s in
    assert (forall (i:nat). i < Seq.length terms ==>
      Seq.index terms i >=. 0.0R);
    rsum_nonnegative terms

open Kuiper.Scalars
module Seq = FStar.Seq

let frobenius_result_length
  (#t:Type0) {| scalar t |}
  (inv : t)
  (#n : nat)
  (s : Seq.lseq t n)
  : Lemma (Seq.length (frobenius_result inv s) == n)
          [SMTPat (Seq.length (frobenius_result inv s))]
  = ()
