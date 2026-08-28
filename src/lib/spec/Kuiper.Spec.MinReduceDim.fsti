module Kuiper.Spec.MinReduceDim

(* Functional specification for min-reduction over the middle
   dimension of a (B, D, M) row-major tensor (KernelBench L1 #53).

       y[b, j] = min_k  x[b, k, j]    for b<B, j<M
       (output shape (B, 1, M) with keepdim=True, or (B, M) without)

   Layout matches [Kuiper.Spec.MaxReduceDim] / [Kuiper.Spec.SumReduceDim]:
   factor (B, D, M) as a 2-D matrix of shape (B*M, D); row r = b*M + j
   carries the length-D slice x[b, :, j].

   [seq_fmin] is a deterministic left fold seeded by +infinity.  This
   spec uses exact equality because the serial kernel performs that same
   fold order; it assumes no global algebraic law for [fmin]. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Math.Fmin
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

let minreduce_post
  (n_rows : nat) (n_cols : nat)
  (sx : chest2 f32 n_rows n_cols)
  (sy : Seq.lseq f32 n_rows)
  : prop =
  forall (r : nat). r < n_rows ==>
    (sy @! r) == seq_fmin (EM.ematrix_row sx r)
