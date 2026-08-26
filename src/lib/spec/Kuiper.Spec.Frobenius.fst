module Kuiper.Spec.Frobenius

open Kuiper.Scalars
module Seq = FStar.Seq

let frobenius_result_length
  (#t:Type0) {| scalar t |}
  (inv : t)
  (#n : nat)
  (s : Seq.lseq t n)
  : Lemma (Seq.length (frobenius_result inv s) == n)
          [SMTPat (Seq.length (frobenius_result inv s))]
  = ()
