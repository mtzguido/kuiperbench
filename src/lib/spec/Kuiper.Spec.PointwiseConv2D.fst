module Kuiper.Spec.PointwiseConv2D

(* Implementation of the PointwiseConv2D functional spec.  Mirrors
   [Kuiper.Spec.Conv2D.fst] reduced to a single (input-channel)
   reduction axis. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Spec.Conv2D

let rec __pwconv2d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #h #w : nat)
  (#cout : nat)
  (x : etensor4 et b_n cin h w)
  (weight : chest2 et cout cin)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h)
  (ow : natlt w)
  (to : nat{to <= cin})
  : GTot et (decreases to)
  = if to = 0 then zero
    else (
      let ic : natlt cin = to - 1 in
      add
        (__pwconv2d_single x weight b oc oh ow (to - 1))
        (mul (tacc x b ic oh ow) (acc2 weight oc ic))
    )

let __pwconv2d_single_zero_lemma
  #et #_ #b_n #cin #h #w #cout
  x weight b oc oh ow
  = ()

let __pwconv2d_single_lemma
  #et #_ #b_n #cin #h #w #cout
  x weight b oc oh ow to
  = ()

let lemma_pwconv2d_index
  #et #_ #b_n #cin #h #w #cout
  x weight bias b oc oh ow
  = ()
