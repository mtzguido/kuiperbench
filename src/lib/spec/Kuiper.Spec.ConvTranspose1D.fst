module Kuiper.Spec.ConvTranspose1D

open Kuiper
open Kuiper.Spec.Conv1D
module Seq = FStar.Seq

(* Implementation of the ConvTranspose1D functional spec. *)

let rec __convT1d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (#cout : nat)
  (kk : pos)
  (s : pos) (p : nat) (d : pos)
  (#l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cin cout kk)
  (b : natlt b_n)
  (oc : natlt cout)
  (ol : natlt l_out)
  (to : nat{to <= (if cin = 0 then 0 else cin * kk)})
  : GTot et (decreases to)
  = if to = 0 then zero
    else (
      let i = to - 1 in
      assert (cin > 0);
      let ic : natlt cin = unrank1_ic cin kk i in
      let k_i : natlt kk = unrank1_k cin kk i in
      let num : int = ol + p - k_i * d in
      add
        (__convT1d_single kk s p d x weight b oc ol (to - 1))
        (mul (read_strided_padded_1d x b ic s num)
             (t3acc weight ic oc k_i))
    )

let __convT1d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (#cout : nat)
  (kk : pos)
  (s : pos) (p : nat) (d : pos)
  (#l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cin cout kk)
  (b : natlt b_n) (oc : natlt cout) (ol : natlt l_out)
  : Lemma
    (ensures __convT1d_single kk s p d x weight b oc ol 0 == zero)
    [SMTPat (__convT1d_single kk s p d x weight b oc ol 0)]
  = ()

let __convT1d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n : nat) (cin : pos) (#l_in : nat)
  (#cout : nat)
  (kk : pos)
  (s : pos) (p : nat) (d : pos)
  (#l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cin cout kk)
  (b : natlt b_n) (oc : natlt cout) (ol : natlt l_out)
  (to : pos{to <= cin * kk})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank1_ic cin kk i in
      let k_i = unrank1_k cin kk i in
      let num : int = ol + p - k_i * d in
      __convT1d_single kk s p d x weight b oc ol to ==
      add
        (__convT1d_single kk s p d x weight b oc ol (to - 1))
        (mul (read_strided_padded_1d x b ic s num)
             (t3acc weight ic oc k_i))))
  = ()

let lemma_convT1d_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (#cout : nat)
  (kk : pos)
  (s : pos) (p : nat) (d : pos)
  (l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cin cout kk)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout) (ol : natlt l_out)
  : Lemma (t3acc (convT1d kk s p d l_out x weight bias) b oc ol
           == convT1d_single kk s p d x weight bias b oc ol)
          [SMTPat (t3acc (convT1d kk s p d l_out x weight bias) b oc ol)]
  = ()
