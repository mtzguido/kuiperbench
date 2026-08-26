module Kuiper.KB.Compat.Product

open FStar.Seq
open FStar.Real
open Kuiper
open Kuiper.Approximates
open Kuiper.Seq.Common

let rec prod_is_approx'
  #a {| scalar a, real_like a |}
  (s : seq a) (s' : seq real) (acc : a) (acc' : real)
  : Lemma
      (requires s %~ s' /\ acc %~ acc')
      (ensures seq_fold_left mul acc s %~ seq_fold_left ( *. ) acc' s')
      (decreases Seq.length s)
  = match view_seq s, view_seq s' with
    | SNil, SNil -> ()
    | SCons hd tl, SCons hd' tl' ->
      prod_is_approx' #a tl tl' (mul acc hd) (acc' *. hd')

let prod_is_approx
  #a {| scalar a, real_like a |}
  (s : seq a) (s' : seq real)
  : Lemma
      (requires s %~ s')
      (ensures seq_fold_left mul one s %~ rprod s')
  = prod_is_approx' s s' one 1.0R
