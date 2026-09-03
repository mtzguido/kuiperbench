module Kuiper.Spec.ConvTranspose2D

(* Functional specification for 2D transposed convolution
   (KernelBench L1 cluster -- nn.ConvTranspose2d).

   Form-(a) scatter definition (per-output-pixel sum):

       y[b, oc, oh, ow] = bias[oc]
                        + Σ_{ic, kh, kw}
                              x_strided[b, ic, oh+P_h-kh*D_h,
                                                ow+P_w-kw*D_w]
                              * w[ic, oc, kh, kw]

   where [x_strided] is [x] viewed at stride [(S_h, S_w)] with
   zero-padding outside support: an output tap at numerator
   [(num_h, num_w)] reads [x[b, ic, num_h/S_h, num_w/S_w]] iff
   both numerators are non-negative, divisible by their stride,
   and the resulting indices are in range [0..H_in)] /
   [0..W_in)].  Other taps contribute [zero].

   This matches PyTorch's
       nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                          stride, padding, output_padding,
                          dilation, groups=1, bias)
   in the groups=1 case.  See
   src/kernelbench/level1/CONV_TRANSPOSED_DESIGN.md for the
   grouped-variant strategy.

   Layout
   ------

   - input  x : (B, C_in, H_in, W_in)
   - weight w : (C_in, C_out, K_h, K_w)   (* transposed weight order *)
   - bias     : lseq et C_out
   - output y : (B, C_out, H_out, W_out)

   Output spatial dims follow the standard PyTorch formula:

       H_out = (H_in - 1) * S_h - 2*P_h + D_h*(K_h - 1)
                + output_padding_h + 1
       W_out = (W_in - 1) * S_w - 2*P_w + D_w*(K_w - 1)
                + output_padding_w + 1

   Output_padding does *not* appear in the spec body: the caller
   chooses [h_out], [w_out] to match PyTorch's formula and the
   per-output sum naturally yields zero for any positions whose
   taps fall outside the input support.  The helper
   [convT_out_len_1d] returns [max 0 raw] of the formula as a
   [nat].

   Spec semantics
   --------------

   Mirrors [Kuiper.Spec.Conv2D]: scalar exact spec via
   [add]/[mul]/[zero] from the [scalar] typeclass, left-fold
   over linearised [(ic, kh, kw)] (reusing [unrank_ic],
   [unrank_kh], [unrank_kw] from [Kuiper.Spec.Conv2D]).
   Floating-point implementations may differ in associativity
   and only need to match the result up to [%~]; lift via
   [Kuiper.Approximates.real_like]. *)

open Kuiper
open Kuiper.Spec.Conv2D
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* Output-length helper (per axis).                                   *)
(* ------------------------------------------------------------------ *)

let convT_out_len_1d
  (h_in : nat) (s : pos) (p : nat) (d : pos) (k : pos) (opad : nat)
  : nat
  = let raw : int = (h_in - 1) * s - 2 * p + d * (k - 1) + opad + 1 in
    if raw < 0 then 0 else raw

(* ------------------------------------------------------------------ *)
(* Strided + zero-padded input read.                                  *)
(* ------------------------------------------------------------------ *)

let read_strided_padded_2d
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (b : natlt b_n) (ic : natlt cin)
  (sh : pos) (sw : pos)
  (h_num : int) (w_num : int)
  : GTot et
  = if h_num >= 0 && w_num >= 0
       && h_num % sh = 0 && w_num % sw = 0
       && h_num / sh < h_in && w_num / sw < w_in
    then tacc x b ic (h_num / sh) (w_num / sw)
    else zero

(* ------------------------------------------------------------------ *)
(* Per-output-pixel partial sum.                                      *)
(* ------------------------------------------------------------------ *)

val __convT2d_single
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
  : GTot et

val __convT2d_single_zero_lemma
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

val __convT2d_single_lemma
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

(* ------------------------------------------------------------------ *)
(* Full per-pixel result and full output tensor.                      *)
(* ------------------------------------------------------------------ *)

let convT2d_single
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
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  : GTot et
  = let to : nat = if cin = 0 then 0 else cin * kh * kw in
    add (Seq.index bias oc)
        (__convT2d_single kh kw sh sw ph pw dh dw
                          x weight b oc oh ow to)

let convT2d
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
  : etensor4 et b_n cout h_out w_out
  = mkT4 (fun b oc oh ow ->
      convT2d_single kh kw sh sw ph pw dh dw
                     x weight bias b oc oh ow)

(* Grouped ConvTranspose2D.  PyTorch stores the weight as
   [(groups*cin_pg), cout_pg, kh, kw].  Output channel [oc] selects group
   [oc / cout_pg], and only the corresponding [cin_pg] input channels and
   weight rows participate. *)
let convT2d_group_input
  (#et:Type)
  (#b_n #groups #cin_pg #h_in #w_in : nat)
  (x : etensor4 et b_n (groups * cin_pg) h_in w_in)
  (g : natlt groups)
  : etensor4 et b_n cin_pg h_in w_in
  = mkT4 (fun b ic ih iw -> tacc x b (g * cin_pg + ic) ih iw)

let convT2d_group_weight
  (#et:Type)
  (#groups #cin_pg #cout_pg #kh #kw : nat)
  (weight : etensor4 et (groups * cin_pg) cout_pg kh kw)
  (g : natlt groups)
  : etensor4 et cin_pg cout_pg kh kw
  = mkT4 (fun ic oc khi kwi ->
      tacc weight (g * cin_pg + ic) oc khi kwi)

let convT2d_group_bias
  (#et:Type)
  (#groups #cout_pg : nat)
  (bias : Seq.lseq et (groups * cout_pg))
  (g : natlt groups)
  : GTot (Seq.lseq et cout_pg)
  = Seq.init_ghost cout_pg (fun oc -> Seq.index bias (g * cout_pg + oc))

let convT2d_grouped_single
  (#et:Type) {| scalar et |}
  (#b_n : nat) (groups cin_pg : pos) (#h_in #w_in : nat)
  (cout_pg kh kw sh sw : pos) (ph pw : nat) (dh dw : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n (groups * cin_pg) h_in w_in)
  (weight : etensor4 et (groups * cin_pg) cout_pg kh kw)
  (bias : Seq.lseq et (groups * cout_pg))
  (b : natlt b_n) (oc : natlt (groups * cout_pg))
  (oh : natlt h_out) (ow : natlt w_out)
  : GTot et
  = let g : natlt groups = oc / cout_pg in
    let oc_pg : natlt cout_pg = oc % cout_pg in
    convT2d_single kh kw sh sw ph pw dh dw
      (convT2d_group_input x g)
      (convT2d_group_weight weight g)
      (convT2d_group_bias bias g)
      b oc_pg oh ow

val lemma_convT2d_index
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
