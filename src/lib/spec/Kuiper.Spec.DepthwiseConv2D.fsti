module Kuiper.Spec.DepthwiseConv2D

(* Functional specification for depthwise 2D convolution
   (KernelBench L1 #82..#85 -- nn.Conv2d with groups = in_channels):

       y[b, c, oh, ow] = bias[c]
                       + Σ_{kh, kw}
                            x[b, c,
                              oh*S_h + kh*D_h - P_h,
                              ow*S_w + kw*D_w - P_w]
                            * w[c, kh, kw]

   where:
     - input  x : (B, C,  H_in,  W_in)
     - weight w : (C, K_h, K_w)        -- one filter per channel
     - bias     : lseq et C            -- one scalar per channel
     - per-axis stride / padding / dilation
     - out-of-range input reads return [zero] (zero padding)

   "Depthwise" here means the channel-multiplier is 1, i.e. the
   PyTorch/KernelBench setup [Conv2d(C, C, ..., groups = C)].  In this
   regime each output channel is the 2D convolution of one input
   channel against one (K_h, K_w) filter — no cross-channel sum.  This
   matches all of KernelBench L1 #82..#85 (and the depthwise stage of
   #86).

   The output spatial dimensions are not computed inside the spec;
   they are taken as parameters [h_out], [w_out] and the caller is
   responsible for the standard relation
       h_out = (H_in + 2*P_h - D_h*(K_h - 1) - 1) / S_h + 1
       w_out = (W_in + 2*P_w - D_w*(K_w - 1) - 1) / S_w + 1.

   The per-pixel sum is folded left-to-right over a linearised
   (kh, kw) index, mirroring the convention of
   [Kuiper.Spec.GEMM.__matmul_single] and
   [Kuiper.Spec.Conv2D.__conv2d_single].  Bit-exact at the [scalar et]
   level; an [%~]-style approximation for floating-point implementations
   is recovered via the standard [real_like] / [a_add] / [a_mul] chain
   already in tree, exactly as for [Kuiper.Spec.Conv2D]. *)

open Kuiper
open Kuiper.Spec.Conv2D
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* Linearised (kh, kw) index decomposition.                           *)
(* ------------------------------------------------------------------ *)

(* Decompose a flat index [i < kh*kw] into [(kh_i, kw_i)].            *)
let unrank_dw_kh
  (kh : pos) (kw : pos)
  (i : nat{i < kh * kw})
  : Tot (natlt kh) =
  FStar.Math.Lemmas.lemma_div_le i (kh * kw - 1) kw;
  FStar.Math.Lemmas.cancel_mul_div kh kw;
  FStar.Math.Lemmas.lemma_div_plus (-1) kh kw;
  i / kw

let unrank_dw_kw
  (kh : pos) (kw : pos)
  (i : nat{i < kh * kw})
  : Tot (natlt kw)
  = i % kw

(* ------------------------------------------------------------------ *)
(* Per-output-pixel partial sum.  Mirrors                             *)
(* [Kuiper.Spec.Conv2D.__conv2d_single] but without the (ic) axis:    *)
(* the input/weight channel coordinate is fixed to the output channel *)
(* coordinate [c].                                                    *)
(* ------------------------------------------------------------------ *)

val __dwconv2d_single
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (weight : etensor4 et c_n 1 kh kw)
  (b : natlt b_n)
  (c : natlt c_n)
  (oh : natlt h_out)
  (ow : natlt w_out)
  (to : nat{to <= kh * kw})
  : GTot et

val __dwconv2d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (weight : etensor4 et c_n 1 kh kw)
  (b : natlt b_n)
  (c : natlt c_n)
  (oh : natlt h_out)
  (ow : natlt w_out)
  : Lemma
    (ensures __dwconv2d_single kh kw s_h s_w p_h p_w d_h d_w
               x weight b c oh ow 0 == zero)
    [SMTPat (__dwconv2d_single kh kw s_h s_w p_h p_w d_h d_w
               x weight b c oh ow 0)]

val __dwconv2d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (weight : etensor4 et c_n 1 kh kw)
  (b : natlt b_n)
  (c : natlt c_n)
  (oh : natlt h_out)
  (ow : natlt w_out)
  (to : pos{to <= kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let kh_i = unrank_dw_kh kh kw i in
      let kw_i = unrank_dw_kw kh kw i in
      let h_idx : int = oh * s_h + kh_i * d_h - p_h in
      let w_idx : int = ow * s_w + kw_i * d_w - p_w in
      __dwconv2d_single kh kw s_h s_w p_h p_w d_h d_w
        x weight b c oh ow to ==
      add
        (__dwconv2d_single kh kw s_h s_w p_h p_w d_h d_w
          x weight b c oh ow (to - 1))
        (mul (read_padded x b c h_idx w_idx)
             (tacc weight c 0 kh_i kw_i))))

(* ------------------------------------------------------------------ *)
(* Full per-pixel result and full output tensor.                      *)
(* ------------------------------------------------------------------ *)

let dwconv2d_single
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (weight : etensor4 et c_n 1 kh kw)
  (bias : Seq.lseq et c_n)
  (b : natlt b_n)
  (c : natlt c_n)
  (oh : natlt h_out)
  (ow : natlt w_out)
  : GTot et
  = add (Seq.index bias c)
        (__dwconv2d_single kh kw s_h s_w p_h p_w d_h d_w
           x weight b c oh ow (kh * kw))

let dwconv2d
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (h_out w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (weight : etensor4 et c_n 1 kh kw)
  (bias : Seq.lseq et c_n)
  : etensor4 et b_n c_n h_out w_out
  = mkT4 (fun b c oh ow ->
      dwconv2d_single kh kw s_h s_w p_h p_w d_h d_w
        x weight bias b c oh ow)

val lemma_dwconv2d_index
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (h_out w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (weight : etensor4 et c_n 1 kh kw)
  (bias : Seq.lseq et c_n)
  (b : natlt b_n) (c : natlt c_n)
  (oh : natlt h_out) (ow : natlt w_out)
  : Lemma (tacc (dwconv2d kh kw s_h s_w p_h p_w d_h d_w
                          h_out w_out x weight bias) b c oh ow
           == dwconv2d_single kh kw s_h s_w p_h p_w d_h d_w
                              x weight bias b c oh ow)
          [SMTPat (tacc (dwconv2d kh kw s_h s_w p_h p_w d_h d_w
                                  h_out w_out x weight bias) b c oh ow)]
