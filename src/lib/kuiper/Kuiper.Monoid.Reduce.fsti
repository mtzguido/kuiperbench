module Kuiper.Monoid.Reduce

(* Reduction-style commutative monoid abstraction.
 *
 * The existing [Kuiper.Monoid.monoid0] is *append-style* (free
 * monoids over [list]/[Seq.seq]) and is not appropriate for
 * fold-style reductions over scalar values where we need the
 * algebraic properties (associativity, commutativity, neutrality)
 * to reorder partial reductions across threads.
 *
 * This module bundles a reduction operator together with a proof
 * that it is a commutative monoid.  Two named instances are
 * provided for the [f32] type:
 *
 *   - [cmonoid_fmax_f32] : reduction by IEEE-754 [fmax] with
 *     identity [-infinity]-shaped via the spec-side [option t]
 *     accumulator pattern.  At the value level the carrier is
 *     simply [f32], with the convention that the caller establishes
 *     the seed by reading the first element rather than starting
 *     from [zero].  We expose [rid = zero] but the kernel uses
 *     this monoid only on non-empty windows where the first read
 *     overwrites the seed; the [rneut] field is therefore stated
 *     for [zero] under the assumption that [fmax zero x] is
 *     well-defined and equal to [x] for [x >= zero] (true in
 *     particular for the squared-magnitude inputs typical of
 *     pooling but not relied on by clients of this module — see
 *     [windowreduce_max_f32] which uses the [option] pattern).
 *
 *   - [cmonoid_fadd_f32] : reduction by IEEE-754 [+].  This is the
 *     foundational case where associativity *fails* on [f32] in
 *     the strict bit-exact sense, so the [rassoc] field is
 *     consumed only by clients that have already lifted to an
 *     approximate [%~]-style postcondition (e.g.
 *     [windowreduce_plus_f32] / [avgpool1d_post]).  Treating it as
 *     a strict monoid here is consistent with how the rest of the
 *     Kuiper stack axiomatizes FP behavior at the type-class
 *     instance boundary (cf. [Kuiper.Approximates.F32],
 *     [Kuiper.Real.exp_log] and friends).
 *
 * Both instances are declared via the project's standard
 * [val] / interface-only idiom — there is no companion .fst.
 * They must be threaded explicitly to consumers (no
 * [tcinstance] tag), since two reduction monoids on the same
 * carrier would be ambiguous to type-class resolution. *)

open Kuiper.Functions
open Kuiper.IntAliases

(* A reduction monoid over carrier [t] is a quadruple of an
 * identity, a binary operator, and proofs that the operator is
 * associative, neutral with respect to the identity, and
 * commutative.  The fields are squashed lemmas rather than
 * computational predicates so the record can be used in [GTot]
 * positions and ghost code without triggering effect issues. *)
noeq
type cmonoid (t : Type) = {
  rid    : t;
  rop    : t -> t -> t;
  rassoc : squash (is_associative rop);
  rneut  : squash (is_neutral_for rid rop);
  rcomm  : squash (is_commutative rop);
}

(* Reduce a (left-)fold of [m.rop] over a [Seq.seq t], starting
 * from accumulator [acc].  This is the canonical fold the kernel
 * proof maintains as an invariant. *)
let red_fold (#t:Type) (m : cmonoid t) (acc : t) (s : FStar.Seq.seq t)
  : GTot t
  = Kuiper.Seq.Common.seq_fold_left m.rop acc s

(* Splitting a fold across an append-point is a direct consequence
 * of [is_monoid] on [(rid, rop)] together with the existing
 * [Kuiper.Seq.Common.lemma_seq_fold_left_sum].  Re-exposed here so
 * clients of [cmonoid] do not have to reach for the underlying
 * [Functions.is_monoid] predicate. *)
val red_fold_append
  (#t:Type) (m : cmonoid t)
  (s1 s2 : FStar.Seq.seq t)
  : Lemma (red_fold m m.rid (FStar.Seq.append s1 s2)
           == m.rop (red_fold m m.rid s1) (red_fold m m.rid s2))

(* The two [f32] instances live in [Kuiper.Monoid.Reduce.F32], which
 * is interface-only following the [Kuiper.Approximates.F32]
 * axiomatic-instance pattern. *)
