module Kuiper.Math.Fmin

(* Law-free [fmin] reduction infrastructure for [f32].
 *
 * [seq_fmin] specifies the exact left-to-right operation order used by the
 * serial min kernels.  No global associativity, commutativity, or infinity
 * neutral law is assumed; the latter would be false for NaNs. *)

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
module Seq = FStar.Seq
module SC  = Kuiper.Seq.Common

(* Fold seed.  This is not claimed to be neutral over NaNs. *)
inline_for_extraction noextract
let pos_inf : f32 = infinity

(* Fold-fmin over a sequence, with [pos_inf] as the seed. *)
let seq_fmin (s : Seq.seq f32) : GTot f32 =
  SC.seq_fold_left fmin pos_inf s

let seq_fmin_empty (_ : unit) : Lemma (seq_fmin Seq.empty == pos_inf) =
  ()

val seq_fmin_snoc (s : Seq.seq f32) (x : f32)
  : Lemma (seq_fmin (Seq.snoc s x) == fmin (seq_fmin s) x)

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
    Seq.lemma_eq_intro pref1 (Seq.snoc pref elt);
    seq_fmin_snoc pref elt
