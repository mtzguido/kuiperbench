module Kuiper.Spec.PointwiseConv2D

(* Functional specification for pointwise 2D convolution
   (KernelBench L1 #87, and the pointwise stage of #86):

       y[b, oc, h, w] = bias[oc]
                      + Σ_{ic = 0}^{C_in - 1} w[oc, ic] * x[b, ic, h, w]

   This is exactly [nn.Conv2d(C_in, C_out, kernel_size = 1, stride = 1,
   padding = 0)]: a 1×1 convolution.  The spatial dimensions of input
   and output are identical (no reduction in (h, w) — every input
   pixel contributes to one output pixel at the same spatial
   coordinate).

   Shapes:
     - input  x : (B, C_in,  H, W)
     - weight w : (C_out, C_in)         (no spatial axes)
     - bias     : lseq et C_out
     - output   : (B, C_out, H, W)

   The reduction is a [scalar et] dot product over the [C_in] input
   channels at a fixed (b, h, w), so pointwise conv is exactly a
   per-pixel matmul with weight matrix [w] of shape [C_out × C_in] and
   per-pixel feature vector [x[b, :, h, w]] of length [C_in].

   That equivalence is what lets the Phase-1 implementation be a thin
   orchestrator on top of the existing
   [Kuiper.Kernel.GEMM.BlockTiling2D] kernel: per batch [b], matmul
   of [(C_out, C_in) × (C_in, H*W)] using a reshape view of
   [x[b, :, :, :]] as an [(C_in, H*W)] row-major matrix (which it
   already is in NCHW memory layout).  The spec below pins one
   accumulation order (left-to-right over [ic]) — floating-point
   implementations get an [%~] approximation via the standard
   [real_like] / [a_add] / [a_mul] chain, exactly as in
   [Kuiper.Spec.Conv2D].

   This module reuses the [etensor4] and [tacc] vocabulary from
   [Kuiper.Spec.Conv2D] for [x] / [y], and the existing
   [Kuiper.EMatrix.chest2] vocabulary for the [(C_out, C_in)]
   weight matrix — that is the natural shape that GEMM kernels
   already consume. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Spec.Conv2D
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* Per-output-pixel partial sum, folded over a prefix of [ic].        *)
(* Mirrors [Kuiper.Spec.GEMM.__matmul_single]: the (oc, ic)-indexed   *)
(* dot-product accumulator at fixed (b, h, w).                        *)
(* ------------------------------------------------------------------ *)

val __pwconv2d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #h #w : nat)
  (#cout : nat)
  (x : etensor4 et b_n cin h w)
  (weight : chest2 et cout cin)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h)
  (ow : natlt w)
  (to : nat{to <= cin})
  : GTot et

val __pwconv2d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #h #w : nat)
  (#cout : nat)
  (x : etensor4 et b_n cin h w)
  (weight : chest2 et cout cin)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h)
  (ow : natlt w)
  : Lemma
    (ensures __pwconv2d_single x weight b oc oh ow 0 == zero)
    [SMTPat (__pwconv2d_single x weight b oc oh ow 0)]

val __pwconv2d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #h #w : nat)
  (#cout : nat)
  (x : etensor4 et b_n cin h w)
  (weight : chest2 et cout cin)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h)
  (ow : natlt w)
  (to : pos{to <= cin})
  : Lemma
    (ensures (
      let ic : natlt cin = to - 1 in
      __pwconv2d_single x weight b oc oh ow to ==
      add
        (__pwconv2d_single x weight b oc oh ow (to - 1))
        (mul (tacc x b ic oh ow) (acc2 weight oc ic))))

(* ------------------------------------------------------------------ *)
(* Full per-pixel result and full output tensor.                      *)
(* ------------------------------------------------------------------ *)

let pwconv2d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #h #w : nat)
  (#cout : nat)
  (x : etensor4 et b_n cin h w)
  (weight : chest2 et cout cin)
  (bias : Seq.lseq et cout)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h)
  (ow : natlt w)
  : GTot et
  = add (Seq.index bias oc)
        (__pwconv2d_single x weight b oc oh ow cin)

let pwconv2d
  (#et:Type) {| scalar et |}
  (#b_n #cin #h #w : nat)
  (#cout : nat)
  (x : etensor4 et b_n cin h w)
  (weight : chest2 et cout cin)
  (bias : Seq.lseq et cout)
  : etensor4 et b_n cout h w
  = mkT4 (fun b oc oh ow -> pwconv2d_single x weight bias b oc oh ow)

val lemma_pwconv2d_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #h #w : nat)
  (#cout : nat)
  (x : etensor4 et b_n cin h w)
  (weight : chest2 et cout cin)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h) (ow : natlt w)
  : Lemma (tacc (pwconv2d x weight bias) b oc oh ow
           == pwconv2d_single x weight bias b oc oh ow)
          [SMTPat (tacc (pwconv2d x weight bias) b oc oh ow)]
