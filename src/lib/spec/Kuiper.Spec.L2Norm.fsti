module Kuiper.Spec.L2Norm

(* Functional specification for row-wise L2 normalisation
   (KernelBench L1 #39):

       y[r, j] = x[r, j] / sqrt(sum_k x[r,k]^2)        for each row r.

   The whole-tensor variant of this is Frobenius normalisation, and we
   re-use its direct real-valued helpers row-wise here.  The contract
   does not expose the implementation's floating reduction result or
   reciprocal square root: every output row directly approximates its
   mathematical real normalization.

   As in Kuiper.Spec.Frobenius, the finite-real approximation relation
   cannot describe the infinity/NaN result of normalizing an all-zero
   row.  The public domain therefore requires every row to have a
   positive real squared norm. *)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
module Seq = FStar.Seq

(* Per-row spec predicate: row [r] of [sx'] directly approximates the
   real-valued normalization of row [r] of [sx]. *)
let row_l2_normalized
  (#t:Type0) {| scalar t, real_like t |}
  (#bd:nat)
  (sx sx' : Seq.lseq t bd)
  (off d : nat{off + d <= bd})
  : prop =
  (Seq.slice sx' off (off + d) <: Seq.seq t) %~
    (frobenius_result_r #d
      (to_real_seq (Seq.slice sx off (off + d))) <: Seq.seq real)

(* Inputs on which every row has a finite-real normalization. *)
let l2norm_domain
  (#t:Type0) {| scalar t, real_like t |}
  (b d : nat)
  (sx : Seq.lseq t (b * d))
  : prop =
  forall (r : nat). r < b ==>
    frobenius_sumsq_r
      (to_real_seq (Seq.slice sx (r * d) (r * d + d))) >. 0.0R

(* Whole-tensor spec: every row is L2-normalised. *)
let l2norm_post
  (#t:Type0) {| scalar t, real_like t |}
  (b d : nat)
  (sx sx' : Seq.lseq t (b * d))
  : prop =
  forall (r : nat). r < b ==>
    (let lo : nat = r * d in
     let hi : nat = lo + d in
     hi <= b * d /\
     row_l2_normalized sx sx' lo d)
