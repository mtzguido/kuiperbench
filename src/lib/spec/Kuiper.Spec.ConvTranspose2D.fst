module Kuiper.Spec.ConvTranspose2D

open Kuiper
open Kuiper.Spec.Conv2D
module Seq = FStar.Seq

(* Implementation of the ConvTranspose2D functional spec. *)

let rec __convT2d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cin cout kh kw)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h_out)
  (ow : natlt w_out)
  (to : nat{to <= (if cin = 0 then 0 else cin * kh * kw)})
  : GTot et (decreases to)
  = if to = 0 then zero
    else (
      let i = to - 1 in
      assert (cin > 0);
      let ic : natlt cin = unrank_ic cin kh kw i in
      let kh_i : natlt kh = unrank_kh cin kh kw i in
      let kw_i : natlt kw = unrank_kw cin kh kw i in
      let h_num : int = oh + ph - kh_i * dh in
      let w_num : int = ow + pw - kw_i * dw in
      add
        (__convT2d_single kh kw sh sw ph pw dh dw
                          x weight b oc oh ow (to - 1))
        (mul (read_strided_padded_2d x b ic sh sw h_num w_num)
             (tacc weight ic oc kh_i kw_i))
    )

let __convT2d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cin cout kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  : Lemma
    (ensures __convT2d_single kh kw sh sw ph pw dh dw
                              x weight b oc oh ow 0
             == zero)
    [SMTPat (__convT2d_single kh kw sh sw ph pw dh dw
                              x weight b oc oh ow 0)]
  = ()

let __convT2d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n : nat) (cin : pos) (#h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cin cout kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank_ic cin kh kw i in
      let kh_i = unrank_kh cin kh kw i in
      let kw_i = unrank_kw cin kh kw i in
      let h_num : int = oh + ph - kh_i * dh in
      let w_num : int = ow + pw - kw_i * dw in
      __convT2d_single kh kw sh sw ph pw dh dw
                       x weight b oc oh ow to ==
      add
        (__convT2d_single kh kw sh sw ph pw dh dw
                          x weight b oc oh ow (to - 1))
        (mul (read_strided_padded_2d x b ic sh sw h_num w_num)
             (tacc weight ic oc kh_i kw_i))))
  = ()

let lemma_convT2d_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (h_out w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cin cout kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  : Lemma (tacc (convT2d kh kw sh sw ph pw dh dw h_out w_out
                         x weight bias) b oc oh ow
           == convT2d_single kh kw sh sw ph pw dh dw
                             x weight bias b oc oh ow)
          [SMTPat (tacc (convT2d kh kw sh sw ph pw dh dw h_out w_out
                                 x weight bias) b oc oh ow)]
  = ()
