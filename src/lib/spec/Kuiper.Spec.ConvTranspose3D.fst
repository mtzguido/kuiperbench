module Kuiper.Spec.ConvTranspose3D

open Kuiper
open Kuiper.Spec.Conv3D
module Seq = FStar.Seq

(* Implementation of the ConvTranspose3D functional spec. *)

let rec __convT3d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cin cout kd kh kw)
  (b : natlt b_n)
  (oc : natlt cout)
  (od : natlt d_out)
  (oh : natlt h_out)
  (ow : natlt w_out)
  (to : nat{to <= (if cin = 0 then 0 else cin * kd * kh * kw)})
  : GTot et (decreases to)
  = if to = 0 then zero
    else (
      let i = to - 1 in
      assert (cin > 0);
      let ic : natlt cin = unrank3_ic cin kd kh kw i in
      let kd_i : natlt kd = unrank3_kd cin kd kh kw i in
      let kh_i : natlt kh = unrank3_kh cin kd kh kw i in
      let kw_i : natlt kw = unrank3_kw cin kd kh kw i in
      let d_num : int = od + pd - kd_i * dd in
      let h_num : int = oh + ph - kh_i * dh in
      let w_num : int = ow + pw - kw_i * dw in
      add
        (__convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                          x weight b oc od oh ow (to - 1))
        (mul (read_strided_padded_3d x b ic sd sh sw d_num h_num w_num)
             (t5acc weight ic oc kd_i kh_i kw_i))
    )

let __convT3d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cin cout kd kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  : Lemma
    (ensures __convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                              x weight b oc od oh ow 0
             == zero)
    [SMTPat (__convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                              x weight b oc od oh ow 0)]
  = ()

let __convT3d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n : nat) (cin : pos) (#d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cin cout kd kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin * kd * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank3_ic cin kd kh kw i in
      let kd_i = unrank3_kd cin kd kh kw i in
      let kh_i = unrank3_kh cin kd kh kw i in
      let kw_i = unrank3_kw cin kd kh kw i in
      let d_num : int = od + pd - kd_i * dd in
      let h_num : int = oh + ph - kh_i * dh in
      let w_num : int = ow + pw - kw_i * dw in
      __convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                       x weight b oc od oh ow to ==
      add
        (__convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                          x weight b oc od oh ow (to - 1))
        (mul (read_strided_padded_3d x b ic sd sh sw d_num h_num w_num)
             (t5acc weight ic oc kd_i kh_i kw_i))))
  = ()

let lemma_convT3d_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (d_out h_out w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cin cout kd kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  : Lemma (t5acc (convT3d kd kh kw sd sh sw pd ph pw dd dh dw
                          d_out h_out w_out x weight bias)
                 b oc od oh ow
           == convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                             x weight bias b oc od oh ow)
          [SMTPat (t5acc (convT3d kd kh kw sd sh sw pd ph pw dd dh dw
                                  d_out h_out w_out x weight bias)
                         b oc od oh ow)]
  = ()
