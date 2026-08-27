module Kuiper.Spec.SumReduceDim

(* Functional specification for sum-reduction over the middle dimension
   of a (B, D, M) row-major tensor (KernelBench L1 #47).

       y[b, j] = Σ_k  x[b, k, j]    for b<B, j<M
       (output shape (B, 1, M) with keepdim=True, or (B, M) with keepdim=False)

   In the Kuiper view we factor the (B, D, M) buffer as a 2-D matrix
   of shape (B*M, D) using the [BCMPages] layout with parameters
   (b=B, hw=M, c=D): row r = b*M + j carries the length-D slice
   x[b, :, j].  The scalar [reduce_batched_block] primitive then
   produces, per row r, an f32 that approximates the real-arithmetic
   sum of that row.  The post-condition uses the [%~] approximation
   relation, mirroring [Kuiper.Spec.Frobenius] / [Kuiper.Spec.L2Norm]:
   tree-reduction order in floating point is not bit-determined, so
   each output cell is only pinned up to [%~] of the mathematical
   real-arithmetic sum.

   Edge cases match PyTorch:
     - All-zero input: real sum is 0, output is 0 (or close to).
     - NaN/Inf in input: propagate; spec is satisfied since the
       existential is on the floating-point witness, not its exact
       value. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Approximates
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

(* Whole-tensor post: every row of the output approximates the real
   sum of the corresponding row of the input.  [sx] is the (B*M, D)
   chest2 view of the input; [sy] is the length-(B*M) output. *)
let sumreduce_post
  (#t:Type0) {| scalar t, real_like t |}
  (n_rows : nat) (n_cols : nat)
  (sx : chest2 t n_rows n_cols)
  (sy : Seq.lseq t n_rows)
  : prop =
  forall (r : nat). r < n_rows ==>
    (sy @! r) %~ rsum (to_real_seq (EM.ematrix_row sx r))
