module Kuiper.Spec.CrossEntropyLoss

open Kuiper.Real

let real_cross_entropy_mul
  (batches : pos)
  (num_classes : pos)
  (rp : Seq.lseq real (batches * num_classes))
  (st : Seq.lseq i64 batches)
  : Lemma
      (real_cross_entropy batches num_classes rp st ==
       rsum (real_cross_entropy_terms batches num_classes rp st) *.
         (1.0R /. FStar.Real.of_int batches))
  = ()
