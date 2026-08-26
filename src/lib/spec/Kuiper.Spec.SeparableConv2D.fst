module Kuiper.Spec.SeparableConv2D

(* Functional specification for depthwise-separable 2D convolution
   (KernelBench L1 #86):

       y = pointwise( depthwise(x) )

   - depthwise stage  : nn.Conv2d(C, C, K, ..., groups = C, ...)
                          spec : Kuiper.Spec.DepthwiseConv2D.dwconv2d
   - pointwise stage  : nn.Conv2d(C, C_out, kernel_size = 1, ...)
                          spec : Kuiper.Spec.PointwiseConv2D.pwconv2d

   The two stages have *separate* weight tensors and *separate* bias
   sequences in PyTorch.  The output spatial dimensions are those of
   the depthwise stage (the pointwise stage preserves spatial size).

   This module is pure F* — no Pulse — and contains only the
   compositional definition plus an index lemma.  No new spec
   primitives are introduced; the proof obligations for an
   implementation of #86 reduce to discharging the per-stage
   postconditions of the underlying [DepthwiseConv2D] and
   [PointwiseConv2D] kernels and chaining them via this composition. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Spec.Conv2D
open Kuiper.Spec.DepthwiseConv2D
open Kuiper.Spec.PointwiseConv2D
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* Composition.  The depthwise output has shape (B, C, H_out, W_out); *)
(* it is fed to the pointwise stage which produces (B, C_out, H_out,  *)
(* W_out).                                                            *)
(* ------------------------------------------------------------------ *)

let separable_conv2d
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (#c_out : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (h_out w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (dw_weight : etensor4 et c_n 1 kh kw)
  (dw_bias   : Seq.lseq et c_n)
  (pw_weight : chest2 et c_out c_n)
  (pw_bias   : Seq.lseq et c_out)
  : etensor4 et b_n c_out h_out w_out
  =
  let mid : etensor4 et b_n c_n h_out w_out =
    dwconv2d kh kw s_h s_w p_h p_w d_h d_w h_out w_out x dw_weight dw_bias
  in
  pwconv2d mid pw_weight pw_bias

(* Index lemma: chase the composition to its per-pixel form.          *)
let lemma_separable_conv2d_index
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (#c_out : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (h_out w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (dw_weight : etensor4 et c_n 1 kh kw)
  (dw_bias   : Seq.lseq et c_n)
  (pw_weight : chest2 et c_out c_n)
  (pw_bias   : Seq.lseq et c_out)
  (b : natlt b_n) (oc : natlt c_out)
  (oh : natlt h_out) (ow : natlt w_out)
  : Lemma
    (tacc (separable_conv2d kh kw s_h s_w p_h p_w d_h d_w
                            h_out w_out x dw_weight dw_bias
                            pw_weight pw_bias)
          b oc oh ow
     == pwconv2d_single
          (dwconv2d kh kw s_h s_w p_h p_w d_h d_w h_out w_out
                    x dw_weight dw_bias)
          pw_weight pw_bias b oc oh ow)
  = ()
