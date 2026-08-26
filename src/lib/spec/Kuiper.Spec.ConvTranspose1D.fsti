module Kuiper.Spec.ConvTranspose1D

(* Functional specification for 1D transposed convolution
   (KernelBench L1 #64, #74, #79 -- nn.ConvTranspose1d).

   Form-(a) scatter definition (per-output-pixel sum):

       y[b, oc, ol] = bias[oc]
                    + Σ_{ic, k}
                          x_strided[b, ic, ol + P - k*D]
                          * w[ic, oc, k]

   where [x_strided] is [x] viewed at stride [S] with zero-padding
   outside support: an output tap at numerator [num] reads
   [x[b, ic, num/S]] iff [num >= 0], [num] divisible by [S], and
   [num/S < L_in].  All other taps contribute [zero].

   This matches PyTorch
       nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                          stride, padding, output_padding,
                          dilation, groups=1, bias)
   in the groups=1 case.  See
   src/kernelbench/level1/CONV_TRANSPOSED_DESIGN.md for the
   grouped-variant strategy.

   Layout
   ------

   - input  x : (B, C_in, L_in)
   - weight w : (C_in, C_out, K)        (* note: ConvTranspose
                                            weight order is transposed
                                            compared to nn.Conv1d *)
   - bias     : lseq et C_out
   - output y : (B, C_out, L_out)

   Output length follows the standard PyTorch formula:

       L_out = (L_in - 1) * S - 2*P + D*(K - 1)
                + output_padding + 1

   The spec body does not see [output_padding]: the caller
   computes [l_out] via [convT1d_out_len] (or directly via
   PyTorch's formula) and the spec produces [zero] for any
   output positions whose taps fall outside the input support.

   Spec semantics
   --------------

   Defined at the [scalar et] level using [add]/[mul]/[zero].
   Accumulation is folded left-to-right over a linearised
   [(ic, k)] index, reusing [unrank1_ic] / [unrank1_k] from
   [Kuiper.Spec.Conv1D].  Same exact-spec / approximate-lift
   story as [Kuiper.Spec.Conv1D] / [Kuiper.Spec.Conv2D]. *)

open Kuiper
open Kuiper.Spec.Conv1D
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* Output-length helper.                                              *)
(* ------------------------------------------------------------------ *)

let convT1d_out_len
  (l_in : nat) (s : pos) (p : nat) (d : pos) (k : pos) (opad : nat)
  : nat
  = let raw : int = (l_in - 1) * s - 2 * p + d * (k - 1) + opad + 1 in
    if raw < 0 then 0 else raw

(* ------------------------------------------------------------------ *)
(* Strided + zero-padded input read.                                  *)
(* ------------------------------------------------------------------ *)

let read_strided_padded_1d
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (x : etensor3 et b_n cin l_in)
  (b : natlt b_n) (ic : natlt cin)
  (s : pos)
  (num : int)
  : GTot et
  = if num >= 0 && num % s = 0 && num / s < l_in
    then t3acc x b ic (num / s)
    else zero

(* ------------------------------------------------------------------ *)
(* Per-output-pixel partial sum.                                      *)
(* ------------------------------------------------------------------ *)

val __convT1d_single
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
  : GTot et

val __convT1d_single_zero_lemma
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

val __convT1d_single_lemma
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

(* ------------------------------------------------------------------ *)
(* Full per-pixel result and full output tensor.                      *)
(* ------------------------------------------------------------------ *)

let convT1d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (#cout : nat)
  (kk : pos)
  (s : pos) (p : nat) (d : pos)
  (#l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cin cout kk)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout) (ol : natlt l_out)
  : GTot et
  = let to : nat = if cin = 0 then 0 else cin * kk in
    add (Seq.index bias oc)
        (__convT1d_single kk s p d x weight b oc ol to)

let convT1d
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (#cout : nat)
  (kk : pos)
  (s : pos) (p : nat) (d : pos)
  (l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cin cout kk)
  (bias : Seq.lseq et cout)
  : etensor3 et b_n cout l_out
  = mkT3 (fun b oc ol ->
      convT1d_single kk s p d x weight bias b oc ol)

val lemma_convT1d_index
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
