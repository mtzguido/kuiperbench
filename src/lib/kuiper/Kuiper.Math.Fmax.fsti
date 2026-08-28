module Kuiper.Math.Fmax

(* Narrow [fmax] reduction infrastructure for [f32].
 *
 * This module is single-purpose: it backs the max-reduction
 * primitive used by KernelBench L1 #49 [torch.max(x, dim=1)].
 *
 * IEEE-754 [fmax] is, on non-NaN inputs, both associative and
 * commutative *bit-exactly* — unlike [+] on the same carrier.
 * That is why the max-reduction kernel post-condition can be
 * stated as exact equality (no [%~] approximation), in contrast
 * to the sum-reduction primitive which must allow for
 * floating-point reassociation slop.
 *
 * The associativity / commutativity / neutrality of [fmax] over
 * the abstract [Kuiper.Float32.t] are stated here as axioms in
 * the same idiom as [Kuiper.Float32.add_comm] and the lemmas in
 * [Kuiper.Approximates.F32].  The identity element [neg_inf] is
 * exposed as an extractable [val] (printed as [(-INFINITY)] by
 * Karamel via [Float32.of_literal]).
 *
 * The user-facing API is the sequence reducer [seq_fmax] together
 * with three lemmas — [seq_fmax_empty], [seq_fmax_singleton],
 * [seq_fmax_append] — which suffice to drive a tree-reduction
 * proof in shared memory. *)

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating
open Kuiper.Float32
open Kuiper.Functions
module Seq = FStar.Seq
module SC  = Kuiper.Seq.Common

(* ── Axioms over the abstract [f32] carrier ───────────────────────────── *)

(* Identity element for [fmax].  Extracted as the IEEE-754
   [-infinity] literal.  Defined concretely via [Float32.of_literal]
   so the reducer_fmax_f32 instance composes cleanly through Karamel. *)
inline_for_extraction noextract
let neg_inf : f32 = neg infinity

val fmax_assoc : squash (is_associative (fmax #f32))
val fmax_comm  : squash (is_commutative (fmax #f32))
val fmax_neg_inf_neutral : squash (is_neutral_for neg_inf (fmax #f32))

(* Bundled monoid fact, derivable from the axioms above. *)
let fmax_is_monoid (_ : unit) : Lemma (is_monoid neg_inf (fmax #f32)) =
  let _ = fmax_assoc in
  let _ = fmax_neg_inf_neutral in
  ()

(* ── Sequence reduction ───────────────────────────────────────────────── *)

(* Fold-fmax over a sequence, with [neg_inf] as the seed. *)
let seq_fmax (s : Seq.seq f32) : GTot f32 =
  SC.seq_fold_left fmax neg_inf s

let seq_fmax_empty (_ : unit) : Lemma (seq_fmax Seq.empty == neg_inf) =
  (* By computation: [seq_fold_left] on [Seq.empty] returns the
     accumulator unchanged, since [view_seq Seq.empty = SNil]. *)
  ()

let seq_fmax_singleton (x : f32)
  : Lemma (seq_fmax (Seq.create 1 x) == x)
  =
    let _ = fmax_neg_inf_neutral in
    let s = Seq.create 1 x in
    assert (Seq.length s == 1);
    assert (Seq.equal (Seq.tail s) Seq.empty);
    (* [seq_fold_left fmax neg_inf s
        = seq_fold_left fmax (fmax neg_inf x) (tail s)
        = seq_fold_left fmax x Seq.empty
        = x] by neutrality. *)
    ()

let seq_fmax_append (s1 s2 : Seq.seq f32)
  : Lemma (seq_fmax (Seq.append s1 s2) == fmax (seq_fmax s1) (seq_fmax s2))
  =
    fmax_is_monoid ();
    SC.lemma_seq_fold_left_sum neg_inf fmax s1 s2

(* The fmax analogue of [Kuiper.Kernel.HReduce.rsum_seq_stride_step]
   — the per-stride accumulation step required by the per-thread tree
   reduction on a strided window of an input row.

   Unlike its [rsum] counterpart (which is admitted in HReduce.fst:399
   pending a [seq_stride] lemma), this version is *fully proved* from
   [seq_fmax_append] + [seq_fmax_singleton] + [Seq.append (seq_take k s)
   (Seq.create 1 (s @! k)) == seq_take (k+1) s].

   This is the only sequence-level fact a future
   [Kuiper.Kernel.HReduce.Block.Max] clone would need beyond
   [seq_fmax_append].  Landing it ahead of time means the kernel clone
   does not need to pay the same admitted-lemma debt as the sum
   version.
*)
let seq_fmax_stride_step
  (s : Seq.seq f32)
  (stride : pos)
  (off : nat{off < stride})
  (k : nat)
  : Lemma (requires k < SC.seq_stride_length s stride off /\
                    k * stride + off < Seq.length s)
          (ensures
            fmax (seq_fmax (SC.seq_take k (SC.seq_stride s stride off)))
                 (Seq.index s (k * stride + off)) ==
            seq_fmax (SC.seq_take (k + 1) (SC.seq_stride s stride off)))
  =
    let strs = SC.seq_stride s stride off in
    let len  = SC.seq_stride_length s stride off in
    (* [strs] has length [len]; index [i] equals [s @! (off + i*stride)]. *)
    assert (Seq.length strs == len);
    assert (k + 1 <= len);
    (* The k-th element of [strs] is [s @! (off + k*stride)]. *)
    assert (Seq.index strs k == Seq.index s (off + k * stride));
    (* [seq_take (k+1) strs] = [seq_take k strs] ++ [strs[k]]. *)
    let pref  = SC.seq_take k strs in
    let elt   = Seq.index strs k in
    let pref1 = SC.seq_take (k + 1) strs in
    Seq.lemma_eq_intro pref1 (Seq.append pref (Seq.create 1 elt));
    (* Combine the two monoid facts. *)
    seq_fmax_append pref (Seq.create 1 elt);
    seq_fmax_singleton elt
