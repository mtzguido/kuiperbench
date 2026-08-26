module Kuiper.KB.Compat.Product

open FStar.Seq
open FStar.Real
open Kuiper
open Kuiper.Approximates
open Kuiper.Seq.Common

let rprod (s : seq real) : real = seq_fold_left ( *. ) 1.0R s

val prod_is_approx'
  #a {| scalar a, real_like a |}
  (s : seq a) (s' : seq real) (acc : a) (acc' : real)
  : Lemma
      (requires s %~ s' /\ acc %~ acc')
      (ensures seq_fold_left mul acc s %~ seq_fold_left ( *. ) acc' s')

val prod_is_approx
  #a {| scalar a, real_like a |}
  (s : seq a) (s' : seq real)
  : Lemma
      (requires s %~ s')
      (ensures seq_fold_left mul one s %~ rprod s')

inline_for_extraction let () = ()
