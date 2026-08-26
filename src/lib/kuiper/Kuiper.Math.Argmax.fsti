module Kuiper.Math.Argmax

(* Narrow axiomatic infrastructure connecting [Float32.gt] to [fmax]
 * and to first-element argmax for the argmax-reduction primitive
 * (KernelBench L1 #51 / #52).
 *
 * The Kuiper [f32] carrier ([Kuiper.Float32.t]) is fully abstract: no
 * axioms in the base layer connect [gt v w], [fmax v w], or
 * propositional equality [v == w] in any way.  Without such a
 * connection, an *implementation* using [gt] for branching cannot
 * be related to a *spec* phrased in terms of [seq_fmax].
 *
 * This module adds the IEEE-754 facts needed to verify the
 * one-thread-per-row argmax kernel in [Kuiper.Kernel.HReduce.Argmax].
 * Each [val] is a plain admitted axiom at the .fsti boundary; all are
 * sound for IEEE-754 [fmaxf] / [Float32.gt] = [>] on non-NaN inputs.
 *
 * NaN caveat: KernelBench inputs are [torch.rand(...)] (no NaNs); on
 * NaN inputs, both PyTorch and these axioms become unspecified.  This
 * is the same implicit assumption as in [Kuiper.Math.Fmax].
 *
 * Symmetric axioms for [Float32.lt] and [Kuiper.Math.Fmin] are
 * provided for the argmin kernel. *)

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Math.Fmax
open Kuiper.Math.Fmin
module Seq = FStar.Seq

(* ── Strict-greater-than ↔ fmax ─────────────────────────────────────── *)

val gt_iff_fmax_strict (v w : f32)
  : Lemma (gt v w == true <==> (fmax v w == v /\ ~(v == w)))

val not_gt_fmax_keeps (v w : f32)
  : Lemma (gt v w == false ==> fmax v w == w)

(* Every value is either strictly greater than [neg_inf] or equal to
   it.  Sound: [-inf] is the unique minimum on non-NaN [f32]. *)
val gt_neg_inf_or_eq (v : f32)
  : Lemma (gt v neg_inf == true \/ v == neg_inf)

(* Every element of a sequence is [<=] the [seq_fmax].  Stated
   contrapositively as [gt _ _ == false] to bypass the missing
   abstract [<=] connection. *)
val seq_fmax_geq (s : Seq.seq f32) (i : nat)
  : Lemma (requires i < Seq.length s)
          (ensures  gt (Seq.index s i) (seq_fmax s) == false)

(* ── Strict-less-than ↔ fmin (symmetric) ─────────────────────────── *)

val lt_iff_fmin_strict (v w : f32)
  : Lemma (lt v w == true <==> (fmin v w == v /\ ~(v == w)))

val not_lt_fmin_keeps (v w : f32)
  : Lemma (lt v w == false ==> fmin v w == w)

val lt_pos_inf_or_eq (v : f32)
  : Lemma (lt v pos_inf == true \/ v == pos_inf)

val seq_fmin_leq (s : Seq.seq f32) (i : nat)
  : Lemma (requires i < Seq.length s)
          (ensures  lt (Seq.index s i) (seq_fmin s) == false)
