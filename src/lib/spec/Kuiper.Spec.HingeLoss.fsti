module Kuiper.Spec.HingeLoss

(* Functional spec for KernelBench L1 #100: broadcast Hinge Loss.

       L = mean_{i,j} max(0, 1 - predictions[i,j] * targets[j])

   [predictions] is a [B x N] matrix and [targets] is a length-[N]
   vector.  The real-valued terms below state the broadcast explicitly:
   every row reads the same [targets[j]]. *)

open Kuiper
open Kuiper.Real
open Kuiper.Chest
open Kuiper.Tensor.Layout { to_seq }
open Kuiper.Tensor.Layout.Alg { l2_col_major }
module Seq = FStar.Seq

let real_hinge_step (p t : real) : real =
  rmax 0.0R (1.0R -. p *. t)

let real_hinge_matrix
  (b n : pos)
  (rp : chest2 real b n)
  (rt : chest1 real n)
  : chest2 real n b
  = mk2 (fun j i ->
      real_hinge_step (acc2 rp i j) (acc1 rt j))

let real_hinge_terms
  (b n : pos)
  (rp : chest2 real b n)
  (rt : chest1 real n)
  : GTot (Seq.lseq real (b * n))
  = to_seq (l2_col_major n b) (real_hinge_matrix b n rp rt)

let real_hinge_broadcast
  (b n : pos)
  (rp : chest2 real b n)
  (rt : chest1 real n)
  : GTot real
  = rsum (real_hinge_terms b n rp rt) /.
      FStar.Real.of_int (b * n)
