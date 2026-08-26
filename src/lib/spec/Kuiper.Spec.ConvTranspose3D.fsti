module Kuiper.Spec.ConvTranspose3D

(* Functional specification for 3D transposed convolution
   (KernelBench L1 #58, #61, #68, #70, #72, #73, #77 --
    nn.ConvTranspose3d).

   Form-(a) scatter definition (per-output-pixel sum):

       y[b, oc, od, oh, ow] = bias[oc]
                            + Σ_{ic, kd, kh, kw}
                                  x_strided[b, ic,
                                            od + P_d - kd*D_d,
                                            oh + P_h - kh*D_h,
                                            ow + P_w - kw*D_w]
                                  * w[ic, oc, kd, kh, kw]

   where [x_strided] is [x] viewed at per-axis stride and
   zero-padding outside support.

   Layout
   ------

   - input  x : (B, C_in, D_in, H_in, W_in)
   - weight w : (C_in, C_out, K_d, K_h, K_w)   (* transposed weight order *)
   - bias     : lseq et C_out
   - output y : (B, C_out, D_out, H_out, W_out)

   Output spatial dims follow the standard PyTorch formula
   per axis (use [Kuiper.Spec.ConvTranspose2D.convT_out_len_1d]
   on each of D / H / W).

   Spec semantics
   --------------

   Mirrors [Kuiper.Spec.Conv3D]: scalar exact spec via
   [add]/[mul]/[zero], left-fold over linearised
   [(ic, kd, kh, kw)] using [unrank3_ic] / [unrank3_kd] /
   [unrank3_kh] / [unrank3_kw] from [Kuiper.Spec.Conv3D].
   Floating-point approximation lifts via
   [Kuiper.Approximates]. *)

open Kuiper
open Kuiper.Spec.Conv3D
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* Strided + zero-padded input read.                                  *)
(* ------------------------------------------------------------------ *)

let read_strided_padded_3d
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (b : natlt b_n) (ic : natlt cin)
  (sd : pos) (sh : pos) (sw : pos)
  (d_num : int) (h_num : int) (w_num : int)
  : GTot et
  = if d_num >= 0 && h_num >= 0 && w_num >= 0
       && d_num % sd = 0 && h_num % sh = 0 && w_num % sw = 0
       && d_num / sd < d_in && h_num / sh < h_in && w_num / sw < w_in
    then t5acc x b ic (d_num / sd) (h_num / sh) (w_num / sw)
    else zero

(* ------------------------------------------------------------------ *)
(* Per-output-pixel partial sum.                                      *)
(* ------------------------------------------------------------------ *)

val __convT3d_single
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
  : GTot et

val __convT3d_single_zero_lemma
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

val __convT3d_single_lemma
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

(* ------------------------------------------------------------------ *)
(* Full per-pixel result and full output tensor.                      *)
(* ------------------------------------------------------------------ *)

let convT3d_single
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
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  : GTot et
  = let to : nat = if cin = 0 then 0 else cin * kd * kh * kw in
    add (Seq.index bias oc)
        (__convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                          x weight b oc od oh ow to)

let convT3d
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
  : etensor5 et b_n cout d_out h_out w_out
  = mkT5 (fun b oc od oh ow ->
      convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                     x weight bias b oc od oh ow)

val lemma_convT3d_index
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
