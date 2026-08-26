module Kuiper.Spec.RMSNorm

(* Functional specification for row-wise RMS normalisation
   (KernelBench L1 #36):

       sumsq_r = Σ_j x[r,j]^2
       inv_r   = 1 / sqrt( sumsq_r / D + eps )
       y[r,j]  = x[r,j] * inv_r                       for each row r.

   The kernel is implemented on top of three verified primitives:
   [HReduce.reduce_batched] (sum of squares per row, via a [sq_step]
   pre-map), [Map.map_gpu] (per-row [s ↦ rsqrt(s*inv_c + eps)]) and
   [RowScale.row_scale] (in-place per-row scaling).  Because the
   mathematical [1/D] is realised at runtime as a host-precomputed
   [inv_c : t] (typically [1.0 / D]), we expose [inv_c] in the spec
   directly rather than relating it to the integer dimension [D].

   As with the L1/L2/Frobenius/MeanVar specs, each row's [sumsq] and
   [inv] are existentially bound: the device-side reduction only
   approximates the real-valued sum-of-squares (in fact in this kernel
   the per-row reduction is a left-fold, so the value is
   *deterministic*, but it is still only an approximation of the
   real-valued sum), and [rsqrt] is opaque to the spec.  We *can* pin:

     * shape: each row of the output is a uniform scaling of the
       corresponding input row by a single per-row factor [inv_r];
     * value: [inv_r == rsqrt (sumsq_r * inv_c + eps)] and [sumsq_r]
       approximates the real-valued sum-of-squares of input row [r].

   We re-use [frobenius_sumsq_r] and [smul_step] from
   Kuiper.Spec.Frobenius for the real-valued sum-of-squares and the
   pointwise scaling step.

   Edge case (all-zero row): [sumsq_r = 0], so
   [inv_r = rsqrt(eps)] which is finite as long as [eps > 0]; the
   row becomes the all-zero row, matching the PyTorch reference.
   If [eps == 0] the spec is still satisfied with [inv_r = rsqrt 0]
   (which is [+inf] in IEEE-754), and the result is NaN-filled. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Scalars.Ops
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
open Kuiper.Kernel.HReduce
open Kuiper.Chest
module EM = Kuiper.EMatrix

(* Pointwise square; alias of [Kuiper.Scalars.square] kept for the
   spec lemmas that reference [sq_step] by name. *)
inline_for_extraction
let sq_step (#t:Type0) {| scalar t |} (x : t) : t = square x

(* Per-row predicate: row [r] of [sx'] is the RMS-normalised version
   of row [r] of [sx], for some existentially-bound [sumsq], [inv]
   pair satisfying the math identities above. *)
let row_rmsnormalized
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#b #d : nat)
  (eps inv_c : t)
  (sx sx' : EM.chest2 t b d)
  (r : natlt b)
  : prop =
  exists (sumsq : t) (inv : t).
    sumsq %~ frobenius_sumsq_r (to_real_seq (EM.ematrix_row sx r)) /\
    inv == rsqrt (add (mul sumsq inv_c) eps) /\
    (forall (j : nat). j < d ==>
       acc2 sx' r j == mul inv (acc2 sx r j))

(* Whole-tensor spec: every row is RMS-normalised. *)
let rmsnorm_post
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (b d : nat)
  (eps inv_c : t)
  (sx sx' : EM.chest2 t b d)
  : prop =
  forall (r : nat). r < b ==> row_rmsnormalized eps inv_c sx sx' r

(* Bridge lemma: the deterministic device-side left-fold of squares
   produced by [reduce_batched sq_step] approximates the real-valued
   mathematical sum-of-squares.  Used to discharge the [sumsq %~ ...]
   conjunct of [row_rmsnormalized] from the exact post-state of
   [reduce_batched]. *)
val row_reduce_partial_sq_approx
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : natlt rows)
  : Lemma (row_reduce_partial (sq_step #t) sx r cols
           %~ frobenius_sumsq_r (to_real_seq (EM.ematrix_row sx r)))
