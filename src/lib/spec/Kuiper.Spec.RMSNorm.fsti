module Kuiper.Spec.RMSNorm

(* Functional specification for row-wise RMS normalisation
   (KernelBench L1 #36):

       sumsq_r = Σ_j x[r,j]^2
       inv_r   = 1 / sqrt( sumsq_r / D + eps )
       y[r,j]  = x[r,j] * inv_r                       for each row r.

   The kernel is implemented on top of three verified primitives:
   [HReduce.reduce_batched] (sum of squares per row, via a [sq_step]
   pre-map), [Map.map_gpu] (per-row [s ↦ rsqrt(s*inv_c + eps)]) and
   [RowScale.row_scale] (in-place per-row scaling).  The mathematical
   [1/D] is computed from the runtime dimension inside the verified
   boundary; it is not supplied by the host or exposed by the contract.

   The public postcondition directly relates each output element to the
   real RMS normalization.  Floating reduction, reciprocal and [rsqrt]
   values are proof-local rather than existentially exposed.

   We re-use [frobenius_sumsq_r] and [smul_step] from
   Kuiper.Spec.Frobenius for the real-valued sum-of-squares and the
   pointwise scaling step.

   Edge case (all-zero row): [sumsq_r = 0], so
   [inv_r = rsqrt(eps)].  Public callers require positive real [eps],
   making this finite and the output the all-zero row. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Scalars.Ops
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
open Kuiper.Kernel.HReduce
open Kuiper.Chest
open Kuiper.Seq.Common { (@!) }
module EM = Kuiper.EMatrix
module RealSqrt = FStar.Math.Sqrt

(* Pointwise square; alias of [Kuiper.Scalars.square] kept for the
   spec lemmas that reference [sq_step] by name. *)
inline_for_extraction
let sq_step (#t:Type0) {| scalar t |} (x : t) : t = square x

let rms_arg_r (#d:pos) (eps:real) (row:Seq.lseq real d) : real =
  frobenius_sumsq_r row /. FStar.Real.of_int d +. eps

let rms_inv_r (#d:pos) (eps:real) (row:Seq.lseq real d) : real =
  let a = rms_arg_r eps row in
  if a >. 0.0R then RealSqrt.rsqrt a else 0.0R

(* Per-row predicate: every output directly approximates the corresponding
   real input multiplied by its real reciprocal RMS. *)
let row_rmsnormalized
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#b : nat) (#d : pos)
  (eps : t)
  (sx sx' : chest2 t b d)
  (r : natlt b)
  : prop =
  let row = to_real_seq (EM.ematrix_row sx r) in
  forall (j : nat). j < d ==>
    acc2 sx' r j %~ ((row @! j) *. rms_inv_r #d (to_real eps) row)

(* Whole-tensor spec: every row is RMS-normalised. *)
let rmsnorm_post
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (b : nat) (d : pos)
  (eps : t)
  (sx sx' : chest2 t b d)
  : prop =
  forall (r : nat). r < b ==> row_rmsnormalized eps sx sx' r

(* Bridge lemma: the deterministic device-side left-fold of squares
   produced by [reduce_batched sq_step] approximates the real-valued
   mathematical sum-of-squares.  Used to discharge the [sumsq %~ ...]
   conjunct of [row_rmsnormalized] from the exact post-state of
   [reduce_batched]. *)
val row_reduce_partial_sq_approx
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (r : natlt rows)
  : Lemma (row_reduce_partial (sq_step #t) sx r cols
           %~ frobenius_sumsq_r (to_real_seq (EM.ematrix_row sx r)))
