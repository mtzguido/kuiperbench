module Kuiper.Spec.ArgmaxReduceDim

(* Functional specification for argmax-reduction over the middle
   dimension of a (B, D, M) row-major tensor (KernelBench L1 #51).

       y[b, j] = argmax_k  x[b, k, j]    for b<B, j<M

   PyTorch [torch.argmax(x, dim=1)] tie-breaking: returns the index
   of the *first* maximum (lowest [k] such that
   [x[b, k, j] == max_k' x[b, k', j]]).  This spec follows that
   convention.

   Layout matches [Kuiper.Spec.MaxReduceDim]: factor (B, D, M) as a
   2-D matrix of shape (B*M, D); row r = b*M + j carries the length-D
   slice x[b, :, j].

   No kernel ships with this spec; see STATUS-narrowing.txt for the
   unblock path and implementation options. *)

open Kuiper
open Kuiper.Math.Fmax
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

(* [k] is the first-max argmax of [s] iff
     - [k < Seq.length s];
     - [s @! k == seq_fmax s];
     - no earlier index has the same value (first-max tie-break). *)
let is_seq_argmax (s : Seq.seq f32) (k : nat) : prop =
  k < Seq.length s /\
  Seq.index s k == seq_fmax s /\
  (forall (i : nat). i < k ==> ~(Seq.index s i == seq_fmax s))

(* Whole-tensor post: every output cell is *the* first-max argmax of
   the corresponding input row.

   The output type is left as [nat] in the spec; the kernel will
   produce a representable [i64] (PyTorch convention) and the bridge
   will witness the bound [sy @! r < n_cols]. *)
let argmaxreduce_post
  (n_rows : nat) (n_cols : nat{n_cols > 0})
  (sx : EM.chest2 f32 n_rows n_cols)
  (sy : Seq.lseq nat n_rows)
  : prop =
  forall (r : nat). r < n_rows ==>
    is_seq_argmax (EM.ematrix_row sx r) (sy @! r)
