module Kuiper.Spec.L2Norm

(* Functional specification for row-wise L2 normalisation
   (KernelBench L1 #39):

       y[r, j] = x[r, j] / sqrt(sum_k x[r,k]^2)        for each row r.

   The whole-tensor variant of this is Frobenius normalisation, and we
   re-use its helpers (in particular [frobenius_result] and
   [frobenius_sumsq_r]) row-wise here.  Each row carries its own
   existentially-bound (inv, sumsq) pair: the device-side tree-reduce
   is non-deterministic up to floating-point rounding, and [rsqrt] is
   opaque to the spec, so we cannot pin [inv] bit-exactly.  We *can*
   pin:

     * shape: each row of the output is a uniform scaling of the
       corresponding input row by a single per-row factor [inv_r];
     * value: [inv_r == rsqrt sumsq_r] and [sumsq_r] approximates the
       real-valued sum-of-squares of input row [r].

   Edge case (all-zero row): see Kuiper.Spec.Frobenius for the same
   discussion -- [rsqrt 0] in IEEE-754 is [+inf] and the row is
   NaN-filled, matching the PyTorch reference.  No precondition on
   non-zero rows is required; the spec is satisfied by the
   degenerate witnesses. *)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
module Seq = FStar.Seq

(* Per-row spec predicate: row [r] of [sx'] is the L2-normalised
   version of row [r] of [sx], for some existentially-bound scaling
   factor [inv_r] approximating the row's reciprocal Frobenius norm. *)
let row_l2_normalized
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#bd:nat)
  (sx sx' : Seq.lseq t bd)
  (off d : nat{off + d <= bd})
  : prop =
  exists (inv : t) (sumsq : t).
    sumsq %~ frobenius_sumsq_r (to_real_seq (Seq.slice sx off (off + d))) /\
    inv == rsqrt sumsq /\
    Seq.slice sx' off (off + d) ==
      frobenius_result #t inv #d (Seq.slice sx off (off + d))

(* Whole-tensor spec: every row is L2-normalised. *)
let l2norm_post
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (b d : nat)
  (sx sx' : Seq.lseq t (b * d))
  : prop =
  forall (r : nat). r < b ==>
    (let lo : nat = r * d in
     let hi : nat = lo + d in
     hi <= b * d /\
     row_l2_normalized sx sx' lo d)
