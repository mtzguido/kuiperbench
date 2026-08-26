module Kuiper.Spec.Conv2DDilated

(* Functional specification for 2D convolution with full per-axis
   (stride, padding, dilation) — generalises [Kuiper.Spec.Conv2D]
   for KernelBench L1 #80 (asymmetric kernel + dilation + padding):

       y[b, oc, oh, ow] = bias[oc]
                        + Σ_{ic, kh, kw}
                              x[b, ic, oh*Sh + kh*Dh - Ph,
                                       ow*Sw + kw*Dw - Pw]
                              * w[oc, ic, kh, kw]

   where:
     - input  x : (B, C_in,  H_in,  W_in)
     - weight w : (C_out, C_in, K_h, K_w)
     - bias     : lseq et C_out
     - per-axis stride (Sh,Sw), padding (Ph,Pw), dilation (Dh,Dw)
     - groups = 1
     - out-of-range input reads return [zero] (zero padding).

   Output spatial dims (h_out, w_out) are parameters; the caller
   is responsible for the standard relation
       h_out = (H_in + 2*Ph - Dh*(K_h-1) - 1) / Sh + 1   (etc.)

   Recovers the [Kuiper.Spec.Conv2D] case with
       Sh = Sw = stride,  Ph = Pw = pad,  Dh = Dw = 1
   (see [lemma_conv2dd_specialises_conv2d] below).

   Reuses [etensor4], [read_padded], [unrank_*] from
   [Kuiper.Spec.Conv2D] -- the index/storage type is identical;
   only the summation rule differs. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Spec.Conv2D
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* Per-output-pixel partial sum.                                      *)
(* ------------------------------------------------------------------ *)

val __conv2dd_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= (if cin = 0 then 0 else cin * kh * kw)})
  : GTot et

val __conv2dd_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  : Lemma
    (ensures __conv2dd_single kh kw sh sw ph pw dh dw
                              x weight b oc oh ow 0 == zero)
    [SMTPat (__conv2dd_single kh kw sh sw ph pw dh dw
                              x weight b oc oh ow 0)]

val __conv2dd_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n : nat) (cin : pos) (#h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank_ic cin kh kw i in
      let kh_i = unrank_kh cin kh kw i in
      let kw_i = unrank_kw cin kh kw i in
      let h_idx : int = oh * sh + kh_i * dh - ph in
      let w_idx : int = ow * sw + kw_i * dw - pw in
      __conv2dd_single kh kw sh sw ph pw dh dw x weight b oc oh ow to ==
      add
        (__conv2dd_single kh kw sh sw ph pw dh dw x weight b oc oh ow (to - 1))
        (mul (read_padded x b ic h_idx w_idx)
             (tacc weight oc ic kh_i kw_i))))

(* ------------------------------------------------------------------ *)
(* Full per-pixel result and full output tensor.                      *)
(* ------------------------------------------------------------------ *)

let conv2dd_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  : GTot et
  = let to : nat = if cin = 0 then 0 else cin * kh * kw in
    add (Seq.index bias oc)
        (__conv2dd_single kh kw sh sw ph pw dh dw x weight b oc oh ow to)

let conv2dd
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (h_out w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (bias : Seq.lseq et cout)
  : etensor4 et b_n cout h_out w_out
  = mkT4 (fun b oc oh ow ->
      conv2dd_single kh kw sh sw ph pw dh dw x weight bias b oc oh ow)

val lemma_conv2dd_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (sh : pos) (sw : pos)
  (ph : nat) (pw : nat)
  (dh : pos) (dw : pos)
  (h_out w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  : Lemma (tacc (conv2dd kh kw sh sw ph pw dh dw h_out w_out x weight bias)
                b oc oh ow
           == conv2dd_single kh kw sh sw ph pw dh dw x weight bias b oc oh ow)
          [SMTPat (tacc (conv2dd kh kw sh sw ph pw dh dw h_out w_out
                                 x weight bias)
                        b oc oh ow)]

(* ------------------------------------------------------------------ *)
(* Specialisation: conv2dd at (sh=sw=s, ph=pw=p, dh=dw=1) is conv2d.  *)
(* ------------------------------------------------------------------ *)

val lemma_conv2dd_specialises_conv2d
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (stride : pos) (pad : nat)
  (h_out w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (bias : Seq.lseq et cout)
  : Lemma
    (ensures
      conv2dd kh kw stride stride pad pad 1 1 h_out w_out x weight bias
        ==
      conv2d kh kw stride pad h_out w_out x weight bias)
