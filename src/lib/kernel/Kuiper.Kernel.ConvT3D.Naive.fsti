module Kuiper.Kernel.ConvT3D.Naive

(* Naive 3D transposed-convolution (ConvTranspose3D) forward kernel:
   one thread per output element.

   Operates on flat NCDHW row-major array1 buffers:
     x      : array1 et (b*cin*d_in*h_in*w_in)
     weight : array1 et (cin*cout*kd*kh*kw)        (* ConvT layout! *)
     bias   : array1 et cout
     y      : array1 et (b*cout*d_out*h_out*w_out)

   Computes (groups=1 case)
     y[b, oc, od, oh, ow] = bias[oc]
        + Σ_{ic, kd_i, kh_i, kw_i}
              x_strided[b, ic,
                        od + pd - kd_i*dd,
                        oh + ph - kh_i*dh,
                        ow + pw - kw_i*dw]
              * weight[ic, oc, kd_i, kh_i, kw_i]

   where [x_strided] reads [x[b, ic, num_d/sd, num_h/sh, num_w/sw]] iff
   the per-axis numerator is non-negative, divisible by its stride, and
   within range.  Other taps contribute zero.

   Matches PyTorch's [nn.ConvTranspose3d] in the groups=1 case, with
   output_padding folded into the [d_out]/[h_out]/[w_out] dims by the
   caller.  Grouped variants are handled at the host (bridge) level by
   slicing channels and calling this kernel once per group.  See
   [Kuiper.Spec.ConvTranspose3D].

   ===
   Same flat-Array1 read/write pattern as Conv3D.Naive / ConvT2D.Naive
   to avoid the krml [__hoisted_0] extraction issue.

   Per-thread post-condition is fully proved via the [convT3d_partial_at]
   loop invariant + [convT3d_partial_at_step] step lemma and a tid-decode
   match (same discharge as Conv3D.Naive; no [assume pure]).  setup/
   teardown/sendability are all discharged at the [kdesc] level with
   [solve] (no [magic ()]). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv3D
open Kuiper.Spec.ConvTranspose3D
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
   [convT3d_single] on the etensor5 induced by the flat input/weight lseqs. *)
let convT3d_out_at
  (#et:Type) {| scalar et |}
  (b cin d_in h_in w_in : nat)
  (cout : nat) (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (d_out h_out w_out : nat)
  (sx : chest1 et (b*cin*d_in*h_in*w_in))
  (sw_lseq : chest1 et (cin*cout*kd*kh*kw))
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
    convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
      (lseq_to_t5 b cin d_in h_in w_in sx)
      (lseq_to_t5 cin cout kd kh kw sw_lseq)
      (chest1_to_seq sbias) bi oc od oh ow

(* The size_t precondition we require from callers. *)
unfold
let convT3d_size_req
  (b cin d_in h_in w_in cout kd kh kw : nat)
  (sd sh sw : nat) (pd ph pw : nat) (dd dh dw : nat)
  (d_out h_out w_out : nat)
  : prop
  = SZ.fits (b * cin * d_in * h_in * w_in) /\
    SZ.fits (cin * cout * kd * kh * kw) /\
    SZ.fits (b * cout * d_out * h_out * w_out) /\
    SZ.fits (cin * kd * kh * kw) /\
    SZ.fits (kd * kh * kw) /\
    SZ.fits (kh * kw) /\
    SZ.fits (h_out * w_out) /\
    SZ.fits (d_out * h_out * w_out) /\
    SZ.fits (cout * d_out * h_out * w_out) /\
    SZ.fits (d_out + pd) /\
    SZ.fits (h_out + ph) /\
    SZ.fits (w_out + pw) /\
    SZ.fits (kd * dd) /\
    SZ.fits (kh * dh) /\
    SZ.fits (kw * dw) /\
    b * cout * d_out * h_out * w_out <= max_blocks * max_threads

inline_for_extraction noextract
fn convt3d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw : sz)
  (dd dh dw d_out h_out w_out : szp)
  (#lx : layout1 (b * cin * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (cin * cout * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*cin*d_in*h_in*w_in)))
  (#sw_l : erased (chest1 et (cin*cout*kd*kh*kw)))
  (#sbias : erased (chest1 et cout))
  (#sy0 : erased (chest1 et (b*cout*d_out*h_out*w_out)))
  (#fx #fw #fb : perm)
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          convT3d_size_req b cin d_in h_in w_in cout kd kh kw
                           sd sh sw pd ph pw dd dh dw
                           d_out h_out w_out)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*d_out*h_out*w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*d_out*h_out*w_out}).
               acc1 sy tid ==
               convT3d_out_at b cin d_in h_in w_in cout kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias tid))
