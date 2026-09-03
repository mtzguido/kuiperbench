module Kuiper.Math.Fmax

(* Law-free [fmax] reduction infrastructure for [f32].
 *
 * [seq_fmax] specifies the exact left-to-right operation order used by the
 * serial max kernels.  In particular, this module does not claim that
 * [fmax] is a monoid over every [f32]: its infinity neutral law is false for
 * NaNs.  The [snoc] lemma below is a structural fact about left folds and
 * requires no floating-point algebra. *)

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating
open Kuiper.Float32
module Seq = FStar.Seq
module SC  = Kuiper.Seq.Common

(* Fold seed.  [of_literal] extracts as the CUDA constant rather than an
   external host symbol.  This is not claimed to be neutral over NaNs. *)
inline_for_extraction noextract
let neg_inf : f32 = of_literal "-INFINITY"

(* Fold-fmax over a sequence, with [neg_inf] as the seed. *)
let seq_fmax (s : Seq.seq f32) : GTot f32 =
  SC.seq_fold_left fmax neg_inf s

let seq_fmax_empty (_ : unit) : Lemma (seq_fmax Seq.empty == neg_inf) =
  ()

val seq_fmax_snoc (s : Seq.seq f32) (x : f32)
  : Lemma (seq_fmax (Seq.snoc s x) == fmax (seq_fmax s) x)

(* The fmax analogue of [Kuiper.Kernel.HReduce.rsum_seq_stride_step]
   — the per-stride accumulation step required by the per-thread tree
   reduction on a strided window of an input row.

   This is proved from [seq_fmax_snoc], without algebraic assumptions.
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
    Seq.lemma_eq_intro pref1 (Seq.snoc pref elt);
    seq_fmax_snoc pref elt
