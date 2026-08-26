module Kuiper.Spec.MinReduceDim

(* Functional specification for min-reduction over the middle
   dimension of a (B, D, M) row-major tensor (KernelBench L1 #53).

       y[b, j] = min_k  x[b, k, j]    for b<B, j<M
       (output shape (B, 1, M) with keepdim=True, or (B, M) without)

   Layout matches [Kuiper.Spec.MaxReduceDim] / [Kuiper.Spec.SumReduceDim]:
   factor (B, D, M) as a 2-D matrix of shape (B*M, D); row r = b*M + j
   carries the length-D slice x[b, :, j].

   IEEE-754 [fmin] on non-NaN inputs is bit-exactly associative+commutative,
   so this spec uses exact sequence equality.

   Two valid implementation routes:
     (a) direct clone of HReduce.Block with [(fmin, pos_inf)] —
         ~700 LOC of Pulse work, parallels the max clone.
     (b) compose: [min x = -max (-x)] — one extra Map kernel.
   Path (b) is much cheaper and matches PyTorch bit-exactly on non-NaN. *)

open Kuiper
open Kuiper.Math.Fmin
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

let minreduce_post
  (n_rows : nat) (n_cols : nat)
  (sx : EM.chest2 f32 n_rows n_cols)
  (sy : Seq.lseq f32 n_rows)
  : prop =
  forall (r : nat). r < n_rows ==>
    (sy @! r) == seq_fmin (EM.ematrix_row sx r)
