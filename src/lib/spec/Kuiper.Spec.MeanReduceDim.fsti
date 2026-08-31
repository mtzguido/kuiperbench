module Kuiper.Spec.MeanReduceDim

(* Functional specification for KernelBench L1 #48: mean-reduction
   along the middle dimension of a (B, D, M) row-major tensor.

       y[b, m] = (1/D) * Σ_d  x[b, d, m]

   The kernel computes the row sum via [reduce_batched_block] and
   post-scales it by the verified in-boundary value [1/D].  The public
   postcondition skips those floating-point intermediates and directly
   states that each output approximates the corresponding real mean. *)

open Kuiper.Common
open Kuiper.Chest
open Kuiper.Scalars
open Kuiper.Real
open Kuiper.Approximates
open Kuiper.Seq.Common { (@!) }
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

(* Per-row predicate: output [r] approximates the mathematical mean of
   input row [r].  [cols] is positive at every kernel call site. *)
let row_mean
  (#t:Type0) {| scalar t, real_like t |}
  (#rows : nat)
  (#cols : nat{cols > 0})
  (sx : chest2 t rows cols)
  (sout' : Seq.lseq t rows)
  (r : natlt rows)
  : prop =
  (sout' @! r) %~
    (rsum (to_real_seq (EM.ematrix_row sx r)) /.
       FStar.Real.of_int cols)

(* Whole-tensor postcondition. *)
let meanreduce_post
  (#t:Type0) {| scalar t, real_like t |}
  (rows : nat)
  (cols : nat{cols > 0})
  (sx : chest2 t rows cols)
  (sout' : Seq.lseq t rows)
  : prop =
  forall (r : nat). r < rows ==> row_mean sx sout' r
