module Kuiper.Spec.Conv3D

open Kuiper
open Kuiper.EMatrix
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality
module Seq = FStar.Seq

(* Implementation of the Conv3D functional spec. *)

let t5acc_pat (#et:Type) (#d0 #d1 #d2 #d3 #d4 : nat)
  (t : etensor5 et d0 d1 d2 d3 d4)
  (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3) (m : natlt d4)
  : Lemma (t5acc t i j k l m == t.f (i, j, k, l, m))
          [SMTPat (t.f (i, j, k, l, m))]
  = ()

let t5_equal #et #d0 #d1 #d2 #d3 #d4
  (t1 t2 : etensor5 et d0 d1 d2 d3 d4) : prop
  = forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3) (m:natlt d4).
      t5acc t1 i j k l m == t5acc t2 i j k l m

let lemma_t5_equal_intro #et #d0 #d1 #d2 #d3 #d4
  (t1 t2 : etensor5 et d0 d1 d2 d3 d4)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2)
                           (l:natlt d3) (m:natlt d4).
                      t5acc t1 i j k l m == t5acc t2 i j k l m)
          (ensures t5_equal t1 t2)
          [SMTPat (t5_equal t1 t2)]
  = ()

let etensor5_ext #et #d0 #d1 #d2 #d3 #d4
  (t1 t2 : etensor5 et d0 d1 d2 d3 d4)
  : Lemma (requires t5_equal t1 t2)
          (ensures t1 == t2)
          [SMTPat (t5_equal t1 t2)]
  = let T5 f1 = t1 in
    let T5 f2 = t2 in
    let aux (idx : natlt d0 & natlt d1 & natlt d2 & natlt d3 & natlt d4)
      : Lemma (f1 idx == f2 idx)
      = let (i, j, k, l, m) = idx in
        assert (t5acc t1 i j k l m == t5acc t2 i j k l m)
    in
    Classical.forall_intro aux;
    F.extensionality_g _ _ f1 f2

let rec __conv3d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (b : natlt b_n)
  (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
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
      let d_idx : int = od * sd + kd_i * dd - pd in
      let h_idx : int = oh * sh + kh_i * dh - ph in
      let w_idx : int = ow * sw + kw_i * dw - pw in
      add
        (__conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                         x weight b oc od oh ow (to - 1))
        (mul (read_padded3 x b ic d_idx h_idx w_idx)
             (t5acc weight oc ic kd_i kh_i kw_i))
    )

let __conv3d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  : Lemma
    (ensures __conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                             x weight b oc od oh ow 0 == zero)
    [SMTPat (__conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                             x weight b oc od oh ow 0)]
  = ()

let __conv3d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n : nat) (cin : pos) (#d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
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
      let d_idx : int = od * sd + kd_i * dd - pd in
      let h_idx : int = oh * sh + kh_i * dh - ph in
      let w_idx : int = ow * sw + kw_i * dw - pw in
      __conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                      x weight b oc od oh ow to ==
      add
        (__conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                         x weight b oc od oh ow (to - 1))
        (mul (read_padded3 x b ic d_idx h_idx w_idx)
             (t5acc weight oc ic kd_i kh_i kw_i))))
  = ()

let lemma_conv3d_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (d_out h_out w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  : Lemma (t5acc (conv3d kd kh kw sd sh sw pd ph pw dd dh dw
                         d_out h_out w_out x weight bias)
                 b oc od oh ow
           == conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                            x weight bias b oc od oh ow)
          [SMTPat (t5acc (conv3d kd kh kw sd sh sw pd ph pw dd dh dw
                                 d_out h_out w_out x weight bias)
                         b oc od oh ow)]
  = ()
