module Kuiper.Spec.TripletMarginLoss

open Kuiper.Real
open Kuiper.Seq.Common
open Kuiper.Seq.Common { (@!) }
module Seq = FStar.Seq

let rec rsum_nonnegative
  (s : Seq.seq real)
  : Lemma
      (requires forall (i : nat). i < Seq.length s ==> Seq.index s i >=. 0.0R)
      (ensures rsum s >=. 0.0R)
      (decreases Seq.length s)
  = match view_seq s with
    | SNil -> ()
    | SCons hd tl ->
      rsum_nonnegative tl;
      rsum_append (Seq.create 1 hd) tl

let real_sq_dist_nonnegative
  (eps : real)
  (d : nat)
  (ra rb : Seq.lseq real d)
  : Lemma (real_sq_dist eps d ra rb >=. 0.0R)
  = let terms = Seq.init d (fun j -> sqdiff_step_r eps (ra @! j) (rb @! j)) in
    assert (forall (i : nat). i < Seq.length terms ==>
      Seq.index terms i >=. 0.0R);
    rsum_nonnegative terms

let real_triplet_loss_mul
  (batches : pos)
  (d : nat)
  (margin eps : real)
  (ra rp rn : Seq.lseq real (batches * d))
  : Lemma
      (real_triplet_loss batches d margin eps ra rp rn ==
       rsum (real_triplet_terms batches d margin eps ra rp rn) *.
         (1.0R /. FStar.Real.of_int batches))
  = ()
