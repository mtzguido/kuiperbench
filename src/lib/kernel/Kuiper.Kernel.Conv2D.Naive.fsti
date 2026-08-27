module Kuiper.Kernel.Conv2D.Naive

(* Naive 2D convolution forward kernel: one thread per output element.

   Operates on flat NCHW row-major array1 buffers:
     x      : array1 et (b*cin*h_in*w_in)
     weight : array1 et (cout*cin*kh*kw)
     bias   : array1 et cout
     y      : array1 et (b*cout*h_out*w_out)

   Computes
     y[b, oc, oh, ow] = bias[oc]
        + Σ_{ic, kh_i, kw_i} x[b, ic, oh*stride + kh_i - pad,
                                       ow*stride + kw_i - pad]
                             * weight[oc, ic, kh_i, kw_i]
   with zero-padded out-of-range reads, dilation = 1, groups = 1.

   The post-condition ties output cells to [Kuiper.Spec.Conv2D.conv2d_single]
   via the canonical row-major flattening [lseq_to_t4]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv2D
module Seq = FStar.Seq
module SZ = Kuiper.SizeT

(* Row-major flattening of an lseq into an [etensor4]. *)
[@@erasable]
val lseq_to_t4
  (#et:Type) (d0 d1 d2 d3 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3))
  : etensor4 et d0 d1 d2 d3

val lseq_to_t4_index
  (#et:Type) (d0 d1 d2 d3 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3))
  (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3)
  : Lemma (tacc (lseq_to_t4 d0 d1 d2 d3 s) i j k l ==
           acc1 s (((i * d1 + j) * d2 + k) * d3 + l))
          [SMTPat (tacc (lseq_to_t4 d0 d1 d2 d3 s) i j k l)]

(* Per-thread post-condition predicate (over output index [tid]).  Decodes
   [tid] into (b, oc, oh, ow) via row-major unflatten, then evaluates
   [conv2d_single] on the etensor4 induced by the flat input/weight lseqs. *)
let conv2d_out_at
  (#et:Type) {| scalar et |}
  (b cin h_in w_in : nat)
  (cout : nat) (kh : pos) (kw : pos)
  (stride : pos) (pad : nat)
  (h_out w_out : nat)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw : chest1 et (cout*cin*kh*kw))
  (sbias : chest1 et cout)
  (tid : nat{tid < b*cout*h_out*w_out})
  : GTot et
  = let bi : natlt b = tid / (cout*h_out*w_out) in
    let r1 = tid % (cout*h_out*w_out) in
    let oc : natlt cout = r1 / (h_out*w_out) in
    let r2 = r1 % (h_out*w_out) in
    let oh : natlt h_out = r2 / w_out in
    let ow : natlt w_out = r2 % w_out in
    conv2d_single kh kw stride pad
      (lseq_to_t4 b cin h_in w_in sx)
      (lseq_to_t4 cout cin kh kw sw)
      (chest1_to_seq sbias) bi oc oh ow

(* The size_t precondition we require from callers.  Encapsulated as a
   single record-shaped prop so signatures stay readable. *)
unfold
let conv2d_size_req
  (b cin h_in w_in cout kh kw : nat)
  (stride : nat)
  (h_out w_out : nat)
  : prop
  = SZ.fits (b * cin * h_in * w_in) /\
    SZ.fits (cout * cin * kh * kw) /\
    SZ.fits (b * cout * h_out * w_out) /\
    SZ.fits (cin * kh * kw) /\
    SZ.fits (h_out * w_out) /\
    SZ.fits (cout * h_out * w_out) /\
    SZ.fits (h_out * stride + kh) /\
    SZ.fits (w_out * stride + kw) /\
    b * cout * h_out * w_out <= max_blocks * max_threads

inline_for_extraction noextract
fn conv2d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout kh kw stride : szp)
  (pad : sz)
  (h_out w_out : szp)
  (#lx : layout1 (b * cin * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*h_out*w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
               acc1 sy tid ==
               conv2d_out_at b cin h_in w_in cout kh kw stride pad
                             h_out w_out sx sw sbias tid))
