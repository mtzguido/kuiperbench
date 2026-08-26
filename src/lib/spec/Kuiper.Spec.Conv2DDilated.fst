module Kuiper.Spec.Conv2DDilated

(* Implementation of Kuiper.Spec.Conv2DDilated.  Mirrors the
   recursive accumulator pattern of [Kuiper.Spec.Conv2D] but with
   per-axis (stride, padding, dilation). *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Spec.Conv2D
module Seq = FStar.Seq

let rec __conv2dd_single
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
  : GTot et (decreases to)
  = if to = 0 then zero
    else (
      let i = to - 1 in
      assert (cin > 0);
      let ic : natlt cin = unrank_ic cin kh kw i in
      let kh_i : natlt kh = unrank_kh cin kh kw i in
      let kw_i : natlt kw = unrank_kw cin kh kw i in
      let h_idx : int = oh * sh + kh_i * dh - ph in
      let w_idx : int = ow * sw + kw_i * dw - pw in
      add
        (__conv2dd_single kh kw sh sw ph pw dh dw x weight b oc oh ow (to - 1))
        (mul (read_padded x b ic h_idx w_idx)
             (tacc weight oc ic kh_i kw_i))
    )

let __conv2dd_single_zero_lemma
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
  = ()

let __conv2dd_single_lemma
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
  = ()

let lemma_conv2dd_index
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
  = ()

(* ------------------------------------------------------------------ *)
(* Pointwise equivalence between [__conv2dd_single] at unit dilation  *)
(* and [__conv2d_single].                                             *)
(* ------------------------------------------------------------------ *)

let rec __conv2dd_eq_conv2d_aux
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (stride : pos) (pad : nat)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= (if cin = 0 then 0 else cin * kh * kw)})
  : Lemma
    (ensures
      __conv2dd_single kh kw stride stride pad pad 1 1 x weight b oc oh ow to
        ==
      Kuiper.Spec.Conv2D.__conv2d_single kh kw stride pad x weight b oc oh ow to)
    (decreases to)
  = if to = 0 then ()
    else (
      assert (cin > 0);
      __conv2dd_eq_conv2d_aux kh kw stride pad x weight b oc oh ow (to - 1);
      let i = to - 1 in
      let ic = unrank_ic cin kh kw i in
      let kh_i = unrank_kh cin kh kw i in
      let kw_i = unrank_kw cin kh kw i in
      // Force unfolding via the per-step lemmas.
      __conv2dd_single_lemma cin kh kw stride stride pad pad 1 1
                             x weight b oc oh ow to;
      Kuiper.Spec.Conv2D.__conv2d_single_lemma cin kh kw stride pad
                                               x weight b oc oh ow to;
      assert (kh_i * 1 == kh_i);
      assert (kw_i * 1 == kw_i);
      let h_idx_d : int = oh * stride + kh_i * 1 - pad in
      let w_idx_d : int = ow * stride + kw_i * 1 - pad in
      let h_idx_n : int = oh * stride + kh_i - pad in
      let w_idx_n : int = ow * stride + kw_i - pad in
      assert (h_idx_d == h_idx_n);
      assert (w_idx_d == w_idx_n)
    )

let lemma_conv2dd_specialises_conv2d
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
  = let lhs = conv2dd kh kw stride stride pad pad 1 1 h_out w_out x weight bias in
    let rhs = conv2d kh kw stride pad h_out w_out x weight bias in
    let aux (b : natlt b_n) (oc : natlt cout)
            (oh : natlt h_out) (ow : natlt w_out)
      : Lemma (tacc lhs b oc oh ow == tacc rhs b oc oh ow)
      = let to : nat = if cin = 0 then 0 else cin * kh * kw in
        __conv2dd_eq_conv2d_aux kh kw stride pad x weight b oc oh ow to
    in
    Classical.forall_intro_4 aux;
    assert (t4_equal lhs rhs)
