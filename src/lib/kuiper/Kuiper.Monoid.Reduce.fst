module Kuiper.Monoid.Reduce

open Kuiper.Functions
open Kuiper.IntAliases
module Seq = FStar.Seq

(* The single non-trivial proof in this module: an [op]/[id] pair
 * that is associative + has [id] as a left+right neutral fits
 * [Kuiper.Functions.is_monoid], and so [seq_fold_left] over
 * [Seq.append] splits on it.  Everything else is interface-only. *)

let red_fold_append
  (#t:Type) (m : cmonoid t)
  (s1 s2 : Seq.seq t)
  : Lemma (red_fold m m.rid (Seq.append s1 s2)
           == m.rop (red_fold m m.rid s1) (red_fold m m.rid s2))
  =
    let _ = m.rassoc in
    let _ = m.rneut in
    assert (is_monoid m.rid m.rop);
    Kuiper.Seq.Common.lemma_seq_fold_left_sum m.rid m.rop s1 s2
