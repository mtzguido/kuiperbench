module Kuiper.Spec.MeanReduceDim

(* Functional specification for KernelBench L1 #48: mean-reduction
   along the middle dimension of a (B, D, M) row-major tensor.

       y[b, m] = (1/D) * Σ_d  x[b, d, m]

   The kernel computes the row sum via [reduce_batched_block] and
   then post-scales by [inv_d : t] (typically [1/D] on the host).
   As in [Kuiper.Spec.RMSNorm], we expose [inv_d] in the spec
   directly rather than relating it to the integer dimension [D].

   Each output cell is the (deterministic) floating-point product of
   a [%~]-approximation [sumr] of the real-arithmetic row sum and the
   externally-supplied scaling factor [inv_d]: i.e., [sout' @! r ==
   mul inv_d sumr] for some [sumr %~ rsum (to_real_seq row)]. *)

open Kuiper.Common
open Kuiper.Scalars
open Kuiper.Real
open Kuiper.Approximates
open Kuiper.Seq.Common { (@!) }
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

(* Per-row predicate: row [r] of [sout'] is the mean of row [r] of
   [sx], modulo the [%~] approximation on the existentially-bound
   floating-point sum. *)
let row_mean
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (inv_d : t)
  (sx : EM.chest2 t rows cols)
  (sout' : Seq.lseq t rows)
  (r : natlt rows)
  : prop =
  exists (sumr : t).
    sumr %~ rsum (to_real_seq (EM.ematrix_row sx r)) /\
    (sout' @! r) == mul inv_d sumr

(* Whole-tensor postcondition. *)
let meanreduce_post
  (#t:Type0) {| scalar t, real_like t |}
  (rows cols : nat)
  (inv_d : t)
  (sx : EM.chest2 t rows cols)
  (sout' : Seq.lseq t rows)
  : prop =
  forall (r : nat). r < rows ==> row_mean inv_d sx sout' r
