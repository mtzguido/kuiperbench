module Kuiper.Kernel.Conv2D.Depthwise

(* Naive 2D depthwise convolution forward kernel: one thread per output element.

   Specialisation of [Kuiper.Kernel.Conv2D.Naive] for the
   PyTorch / KernelBench depthwise regime
       nn.Conv2d(C, C, ..., groups = C)
   where each output channel pulls from its single matching input channel
   only (channel-multiplier 1).

   Operates on flat NCHW row-major array1 buffers:
     x      : array1 et (b*c*h_in*w_in)
     weight : array1 et (c*1*kh*kw)
     bias   : array1 et c
     y      : array1 et (b*c*h_out*w_out)

   Computes
     y[b, c, oh, ow] = bias[c]
        + Σ_{kh_i, kw_i}
            x[b, c, oh*stride + kh_i - pad, ow*stride + kw_i - pad]
            * weight[c, 0, kh_i, kw_i]
   with zero-padded out-of-range reads, dilation = 1, scalar
   stride / pad (same in H and W axes — sufficient for KB L1 #82..#86).

   The post-condition ties output cells to
   [Kuiper.Spec.DepthwiseConv2D.dwconv2d_single] via the canonical
   row-major flattening [Kuiper.Kernel.Conv2D.Naive.lseq_to_t4]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv2D
open Kuiper.Spec.DepthwiseConv2D
open Kuiper.Kernel.Conv2D.Naive
module Seq = FStar.Seq
module SZ = Kuiper.SizeT

(* Per-thread post-condition predicate (over output index [tid]).  Decodes
   [tid] into (b, c, oh, ow) via row-major unflatten, then evaluates
   [dwconv2d_single] on the etensor4 induced by the flat input/weight lseqs. *)
let dwconv2d_out_at
  (#et:Type) {| scalar et |}
  (b c h_in w_in : nat)
  (kh : pos) (kw : pos)
  (stride : pos) (pad : nat)
  (h_out w_out : nat)
  (sx : chest1 et (b*c*h_in*w_in))
  (sw : chest1 et (c*1*kh*kw))
  (sbias : chest1 et c)
  (tid : nat{tid < b*c*h_out*w_out})
  : GTot et
  = let bi : natlt b = tid / (c*h_out*w_out) in
    let r1 = tid % (c*h_out*w_out) in
    let ci : natlt c = r1 / (h_out*w_out) in
    let r2 = r1 % (h_out*w_out) in
    let oh : natlt h_out = r2 / w_out in
    let ow : natlt w_out = r2 % w_out in
    dwconv2d_single kh kw stride stride pad pad 1 1
      (lseq_to_t4 b c h_in w_in sx)
      (lseq_to_t4 c 1 kh kw sw)
      (chest1_to_seq sbias) bi ci oh ow

(* The size_t precondition we require from callers. *)
unfold
let dwconv2d_size_req
  (b c h_in w_in kh kw : nat)
  (stride : nat)
  (h_out w_out : nat)
  : prop
  = SZ.fits (b * c * h_in * w_in) /\
    SZ.fits (c * 1 * kh * kw) /\
    SZ.fits (b * c * h_out * w_out) /\
    SZ.fits (kh * kw) /\
    SZ.fits (h_out * w_out) /\
    SZ.fits (c * h_out * w_out) /\
    SZ.fits (h_out * stride + kh) /\
    SZ.fits (w_out * stride + kw) /\
    b * c * h_out * w_out <= max_blocks * max_threads

inline_for_extraction noextract
fn dwconv2d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b c h_in w_in kh kw stride : szp)
  (pad : sz)
  (h_out w_out : szp)
  (#lx : layout1 (b * c * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (c * 1 * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 c) {| ctlayout lbias |}
  (#ly : layout1 (b * c * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*c*h_in*w_in)))
  (#sw : erased (chest1 et (c*1*kh*kw)))
  (#sbias : erased (chest1 et c))
  (#sy0 : erased (chest1 et (b*c*h_out*w_out)))
  (#fx #fw #fb : perm)
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*c*h_out*w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*c*h_out*w_out}).
               acc1 sy tid ==
               dwconv2d_out_at b c h_in w_in kh kw stride pad
                               h_out w_out sx sw sbias tid))
