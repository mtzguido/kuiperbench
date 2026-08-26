module Kuiper.Spec.ArgminReduceDim

(* Functional specification for argmin-reduction over the middle
   dimension of a (B, D, M) row-major tensor (KernelBench L1 #52).

   Mirror of [Kuiper.Spec.ArgmaxReduceDim] with [seq_fmax] / [fmax]
   replaced by [seq_fmin] / [fmin].

   PyTorch [torch.argmin(x, dim=1)] returns the index of the *first*
   minimum.

   No kernel ships; see STATUS-narrowing.txt for the unblock path. *)

open Kuiper
open Kuiper.Math.Fmin
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

(* [k] is the first-min argmin of [s] iff
     - [k < Seq.length s];
     - [s @! k == seq_fmin s];
     - no earlier index has the same value. *)
let is_seq_argmin (s : Seq.seq f32) (k : nat) : prop =
  k < Seq.length s /\
  Seq.index s k == seq_fmin s /\
  (forall (i : nat). i < k ==> ~(Seq.index s i == seq_fmin s))

let argminreduce_post
  (n_rows : nat) (n_cols : nat{n_cols > 0})
  (sx : EM.chest2 f32 n_rows n_cols)
  (sy : Seq.lseq nat n_rows)
  : prop =
  forall (r : nat). r < n_rows ==>
    is_seq_argmin (EM.ematrix_row sx r) (sy @! r)
