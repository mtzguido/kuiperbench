module Kuiper.Kernel.Conv3D.Naive

(* Naive 3D convolution forward kernel: one thread per output element.

   Operates on flat NCDHW row-major array1 buffers:
     x      : array1 et (b*cin*d_in*h_in*w_in)
     weight : array1 et (cout*cin*kd*kh*kw)
     bias   : array1 et cout
     y      : array1 et (b*cout*d_out*h_out*w_out)

   Computes
     y[b, oc, od, oh, ow] = bias[oc]
        + Σ_{ic, kd_i, kh_i, kw_i}
              x[b, ic, od*stride + kd_i - pad,
                       oh*stride + kh_i - pad,
                       ow*stride + kw_i - pad]
            * weight[oc, ic, kd_i, kh_i, kw_i]
   with zero-padded out-of-range reads, dilation = 1, groups = 1.

   Mirrors [Kuiper.Kernel.Conv2D.Naive] in structure (one extra axis D).
   The post-condition ties output cells to
   [Kuiper.Spec.Conv3D.conv3d_single] via the canonical row-major
   flattening [lseq_to_t5]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv3D
module Seq = FStar.Seq
module SZ = Kuiper.SizeT

(* Row-major flattening of an lseq into an [etensor5]. *)
[@@erasable]
val lseq_to_t5
  (#et:Type) (d0 d1 d2 d3 d4 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3 * d4))
  : etensor5 et d0 d1 d2 d3 d4

val lseq_to_t5_index
  (#et:Type) (d0 d1 d2 d3 d4 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3 * d4))
  (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3) (m:natlt d4)
  : Lemma (t5acc (lseq_to_t5 d0 d1 d2 d3 d4 s) i j k l m ==
           acc1 s ((((i * d1 + j) * d2 + k) * d3 + l) * d4 + m))
          [SMTPat (t5acc (lseq_to_t5 d0 d1 d2 d3 d4 s) i j k l m)]

(* Per-thread post-condition predicate (over output index [tid]).  Decodes
   [tid] into (b, oc, od, oh, ow) via row-major unflatten, then evaluates
   [conv3d_single] on the etensor5 induced by the flat input/weight lseqs.
   Dilation is hard-wired to 1 in the kernel; the spec is fully general
   but the kernel only fixes the dilation=1 instantiation (matching the
   Conv2D.Naive precedent). *)
let conv3d_out_at
  (#et:Type) {| scalar et |}
  (b cin d_in h_in w_in : nat)
  (cout : nat) (kd : pos) (kh : pos) (kw : pos)
  (stride : pos) (pad : nat)
  (d_out h_out w_out : nat)
  (sx : chest1 et (b*cin*d_in*h_in*w_in))
  (sw : chest1 et (cout*cin*kd*kh*kw))
  (sbias : chest1 et cout)
  (tid : nat{tid < b*cout*d_out*h_out*w_out})
  : GTot et
  = let bi : natlt b = tid / (cout*d_out*h_out*w_out) in
    let r1 = tid % (cout*d_out*h_out*w_out) in
    let oc : natlt cout = r1 / (d_out*h_out*w_out) in
    let r2 = r1 % (d_out*h_out*w_out) in
    let od : natlt d_out = r2 / (h_out*w_out) in
    let r3 = r2 % (h_out*w_out) in
    let oh : natlt h_out = r3 / w_out in
    let ow : natlt w_out = r3 % w_out in
    conv3d_single kd kh kw stride stride stride pad pad pad 1 1 1
      (lseq_to_t5 b cin d_in h_in w_in sx)
      (lseq_to_t5 cout cin kd kh kw sw)
      (chest1_to_seq sbias) bi oc od oh ow

(* The size_t precondition we require from callers.  Encapsulated as a
   single record-shaped prop so signatures stay readable. *)
unfold
let conv3d_size_req
  (b cin d_in h_in w_in cout kd kh kw : nat)
  (stride : nat)
  (d_out h_out w_out : nat)
  : prop
  = SZ.fits (b * cin * d_in * h_in * w_in) /\
    SZ.fits (cout * cin * kd * kh * kw) /\
    SZ.fits (b * cout * d_out * h_out * w_out) /\
    SZ.fits (cin * kd * kh * kw) /\
    SZ.fits (kd * kh * kw) /\
    SZ.fits (kh * kw) /\
    SZ.fits (h_out * w_out) /\
    SZ.fits (d_out * h_out * w_out) /\
    SZ.fits (cout * d_out * h_out * w_out) /\
    SZ.fits (d_out * stride + kd) /\
    SZ.fits (h_out * stride + kh) /\
    SZ.fits (w_out * stride + kw) /\
    b * cout * d_out * h_out * w_out <= max_blocks * max_threads

inline_for_extraction noextract
fn conv3d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout kd kh kw stride : szp)
  (pad : sz)
  (d_out h_out w_out : szp)
  (#lx : layout1 (b * cin * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*d_in*h_in*w_in))
  (#sw : chest1 et (cout*cin*kd*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*d_out*h_out*w_out))
  (#fx #fw #fb : perm)
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          conv3d_size_req b cin d_in h_in w_in cout kd kh kw stride
                          d_out h_out w_out)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*d_out*h_out*w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*d_out*h_out*w_out}).
               acc1 sy tid ==
               conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                             d_out h_out w_out sx sw sbias tid))
