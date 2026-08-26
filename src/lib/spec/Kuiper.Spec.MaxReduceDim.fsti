module Kuiper.Spec.MaxReduceDim

(* Functional specification for max-reduction over the middle
   dimension of a (B, D, M) row-major tensor (KernelBench L1 #49).

       y[b, j] = max_k  x[b, k, j]    for b<B, j<M
       (output shape (B, 1, M) with keepdim=True, or (B, M) without)

   Layout matches [Kuiper.Spec.SumReduceDim]: factor (B, D, M) as a
   2-D matrix of shape (B*M, D) using the [BCMPages] layout with
   parameters (b=B, hw=M, c=D); row r = b*M + j carries the length-D
   slice x[b, :, j].  The scalar [reduce_batched_block_max] primitive
   then produces, per row r, an f32 that approximates the real-arithmetic
   max of that row.  The post-condition uses the [%~] approximation
   relation, mirroring [Kuiper.Spec.SumReduceDim]: tree-reduction order
   in floating point is not bit-determined, so each output cell is only
   pinned up to [%~] of the mathematical real-arithmetic max.

   Edge cases match PyTorch:
     - Empty rows are not realised: the spec parameter [n_cols] is
       constrained [> 0], and the kernel only runs at [cols > 0]
       (enforced at the call site by [szp]).
     - NaN/Inf in input: propagate; spec is satisfied since the
       existential is on the floating-point witness, not its exact
       value. *)

open Kuiper
open Kuiper.Approximates
open Kuiper.Math.OnlineSoftmax { seq_max }
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

(* Whole-tensor post: every row of the output approximates the real
   max of the corresponding row of the input.  [sx] is the (B*M, D)
   chest2 view of the input; [sy] is the length-(B*M) output. *)
let maxreduce_post
  (#t:Type0) {| scalar t, real_like t |}
  (n_rows : nat) (n_cols : nat{n_cols > 0})
  (sx : EM.chest2 t n_rows n_cols)
  (sy : Seq.lseq t n_rows)
  : prop =
  forall (r : nat). r < n_rows ==>
    (sy @! r) %~ seq_max (to_real_seq (EM.ematrix_row sx r))
