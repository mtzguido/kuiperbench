module Kuiper.Spec.Scan1D

open Kuiper
open Kuiper.Monoid.Reduce
module Seq = FStar.Seq
module KS  = Kuiper.Seq.Common

(* All four lemmas below are direct consequences of the defining
   equations of [Seq.init_ghost] / [Seq.slice] / [seq_fold_left]
   and need no proof beyond F*'s computational unfolding.  The
   intent of stating them in the .fsti is to expose the per-cell
   identity as an [SMTPat] for downstream specs. *)

let scan_inclusive_length
  (#t:Type) (m : reducer t) (s : Seq.seq t)
  : Lemma (Seq.length (scan_inclusive m s) == Seq.length s)
          [SMTPat (Seq.length (scan_inclusive m s))]
  = ()

let scan_exclusive_length
  (#t:Type) (m : reducer t) (s : Seq.seq t)
  : Lemma (Seq.length (scan_exclusive m s) == Seq.length s)
          [SMTPat (Seq.length (scan_exclusive m s))]
  = ()

let seq_rev_length
  (#a:Type) (s : Seq.seq a)
  : Lemma (Seq.length (seq_rev s) == Seq.length s)
          [SMTPat (Seq.length (seq_rev s))]
  = ()

let scan_inclusive_index
  (#t:Type) (m : reducer t) (s : Seq.seq t) (i : nat { i < Seq.length s })
  : Lemma ((scan_inclusive m s) @! i == scan_inclusive_at m s i)
          [SMTPat ((scan_inclusive m s) @! i)]
  = ()

let scan_exclusive_index
  (#t:Type) (m : reducer t) (s : Seq.seq t) (i : nat { i < Seq.length s })
  : Lemma ((scan_exclusive m s) @! i == scan_exclusive_at m s i)
          [SMTPat ((scan_exclusive m s) @! i)]
  = ()

(* Cell 0 of the exclusive scan equals the reducer seed.
   [Seq.slice s 0 0] is empty, so [red_fold m m.rid empty == m.rid]
   by [seq_fold_left]'s [SNil] arm. *)
let scan_exclusive_zero
  (#t:Type) (m : reducer t) (s : Seq.seq t { Seq.length s > 0 })
  : Lemma ((scan_exclusive m s) @! 0 == m.rid)
  = let empty : Seq.seq t = Seq.slice s 0 0 in
    assert (Seq.equal empty Seq.empty);
    assert (KS.view_seq #t Seq.empty == KS.SNil);
    ()
