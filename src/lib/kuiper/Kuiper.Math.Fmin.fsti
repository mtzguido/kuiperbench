module Kuiper.Math.Fmin

(* Narrow [fmin] reduction infrastructure for [f32].
 *
 * Mirror of [Kuiper.Math.Fmax] for the min-reduction kernel
 * backing KernelBench L1 #53 [torch.min(x, dim=1)].
 *
 * IEEE-754 [fmin] on non-NaN inputs is bit-exact associative and
 * commutative, so the min-reduction post-condition uses exact
 * sequence equality (no [%~]).
 *
 * Identity element is [pos_inf] (+infinity).
 *
 * NOTE: an alternative implementation route for #53 is
 *
 *     min(x) = -max(-x)
 *
 * which reuses the entire [Kuiper.Math.Fmax] / [HReduce.Block.Max]
 * stack at the cost of one extra Map kernel.  This module exists
 * to support a *direct* Block.Min clone if the Map-fusion path
 * proves undesirable for any reason; if path (b) is taken, this
 * file remains useful as the spec-side identity for
 * [Spec.MinReduceDim]. *)

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Functions
module Seq = FStar.Seq
module SC  = Kuiper.Seq.Common

(* ── Axioms over the abstract [f32] carrier ───────────────────────────── *)

(* Identity element for [fmin].  Extracted as the IEEE-754
   [+infinity] literal.  Defined concretely via [Float32.of_literal]
   so the cmonoid_fmin_f32 instance composes cleanly through Karamel. *)
inline_for_extraction noextract
let pos_inf : f32 = infinity

val fmin_assoc : squash (is_associative (fmin #f32))
val fmin_comm  : squash (is_commutative (fmin #f32))
val fmin_pos_inf_neutral : squash (is_neutral_for pos_inf (fmin #f32))

(* Bundled monoid fact, derivable from the axioms above. *)
let fmin_is_monoid (_ : unit) : Lemma (is_monoid pos_inf (fmin #f32)) =
  let _ = fmin_assoc in
  let _ = fmin_pos_inf_neutral in
  ()

(* ── Sequence reduction ───────────────────────────────────────────────── *)

(* Fold-fmin over a sequence, with [pos_inf] as the seed. *)
let seq_fmin (s : Seq.seq f32) : GTot f32 =
  SC.seq_fold_left fmin pos_inf s

let seq_fmin_empty (_ : unit) : Lemma (seq_fmin Seq.empty == pos_inf) =
  ()

let seq_fmin_singleton (x : f32)
  : Lemma (seq_fmin (Seq.create 1 x) == x)
  =
    let _ = fmin_pos_inf_neutral in
    let s = Seq.create 1 x in
    assert (Seq.length s == 1);
    assert (Seq.equal (Seq.tail s) Seq.empty);
    ()

let seq_fmin_append (s1 s2 : Seq.seq f32)
  : Lemma (seq_fmin (Seq.append s1 s2) == fmin (seq_fmin s1) (seq_fmin s2))
  =
    fmin_is_monoid ();
    SC.lemma_seq_fold_left_sum pos_inf fmin s1 s2

(* Mirror of [seq_fmax_stride_step].  Proved cleanly without admits. *)
let seq_fmin_stride_step
  (s : Seq.seq f32)
  (stride : pos)
  (off : nat{off < stride})
  (k : nat)
  : Lemma (requires k < SC.seq_stride_length s stride off /\
                    k * stride + off < Seq.length s)
          (ensures
            fmin (seq_fmin (SC.seq_take k (SC.seq_stride s stride off)))
                 (Seq.index s (k * stride + off)) ==
            seq_fmin (SC.seq_take (k + 1) (SC.seq_stride s stride off)))
  =
    let strs = SC.seq_stride s stride off in
    let len  = SC.seq_stride_length s stride off in
    assert (Seq.length strs == len);
    assert (k + 1 <= len);
    assert (Seq.index strs k == Seq.index s (off + k * stride));
    let pref  = SC.seq_take k strs in
    let elt   = Seq.index strs k in
    let pref1 = SC.seq_take (k + 1) strs in
    Seq.lemma_eq_intro pref1 (Seq.append pref (Seq.create 1 elt));
    seq_fmin_append pref (Seq.create 1 elt);
    seq_fmin_singleton elt
