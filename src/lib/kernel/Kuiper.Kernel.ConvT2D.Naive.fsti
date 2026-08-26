module Kuiper.Kernel.ConvT2D.Naive

(* Naive 2D transposed-convolution (ConvTranspose2D) forward kernel:
   one thread per output element.

   Operates on flat NCHW row-major array1 buffers:
     x      : array1 et (b*cin*h_in*w_in)
     weight : array1 et (cin*cout*kh*kw)        (* ConvT layout! *)
     bias   : array1 et cout
     y      : array1 et (b*cout*h_out*w_out)

   Computes
     y[b, oc, oh, ow] = bias[oc]
        + Σ_{ic, kh_i, kw_i}
            x_strided[b, ic, oh+ph - kh_i*dh, ow+pw - kw_i*dw]
            * weight[ic, oc, kh_i, kw_i]

   where [x_strided] reads [x[b, ic, num_h/sh, num_w/sw]] iff
   [num_h, num_w >= 0], divisible by their stride, and within
   [0..h_in)] / [0..w_in)].  Other taps contribute zero.

   This matches PyTorch's [nn.ConvTranspose2d] in the groups=1 case,
   with output_padding folded into the [h_out]/[w_out] dims by the
   caller.  See [Kuiper.Spec.ConvTranspose2D].

   ===
   Extraction-blocker workaround:  this kernel is structurally
   identical to the prior [~/kuiper-conv-t/.../ConvT2D.Naive], but
   uses the flat Array1 read/write pattern (a la Conv3D) instead of
   Array4 cell reads — krml drops [__hoisted_0] when extracting deep
   [conc(desc d0 d1 d2 d3)]-typed reads, but lowers flat array reads
   cleanly.  See Conv3D.Naive for the prototype.

   Per-thread post-condition is fully proved (no [assume pure]): the
   inner tap accumulation is matched to
   [Kuiper.Spec.ConvTranspose2D.__convT2d_single] via a
   [convT2d_partial_at] loop invariant plus a step lemma, with a
   tid-decode match.  Setup, teardown, and kpre/kpost sendability are
   all discharged at the [kdesc] level with [solve] (no [magic ()]). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv2D
open Kuiper.Spec.ConvTranspose2D
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
   [convT2d_single] on the etensor4 induced by the flat input/weight lseqs. *)
let convT2d_out_at
  (#et:Type) {| scalar et |}
  (b cin h_in w_in : nat)
  (cout : nat) (kh : pos) (kw : pos)
  (sh : pos) (sw : pos) (ph : nat) (pw : nat) (dh : pos) (dw : pos)
  (h_out w_out : nat)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_lseq : chest1 et (cin*cout*kh*kw))
  (sbias : chest1 et cout)
  (tid : nat{tid < b*cout*h_out*w_out})
  : GTot et
  = let bi : natlt b = tid / (cout*h_out*w_out) in
    let r1 = tid % (cout*h_out*w_out) in
    let oc : natlt cout = r1 / (h_out*w_out) in
    let r2 = r1 % (h_out*w_out) in
    let oh : natlt h_out = r2 / w_out in
    let ow : natlt w_out = r2 % w_out in
    convT2d_single kh kw sh sw ph pw dh dw
      (lseq_to_t4 b cin h_in w_in sx)
      (lseq_to_t4 cin cout kh kw sw_lseq)
      (chest1_to_seq sbias) bi oc oh ow

(* The size_t precondition we require from callers. *)
unfold
let convT2d_size_req
  (b cin h_in w_in cout kh kw : nat)
  (sh sw : nat) (ph pw : nat) (dh dw : nat)
  (h_out w_out : nat)
  : prop
  = SZ.fits (b * cin * h_in * w_in) /\
    SZ.fits (cin * cout * kh * kw) /\
    SZ.fits (b * cout * h_out * w_out) /\
    SZ.fits (cin * kh * kw) /\
    SZ.fits (kh * kw) /\
    SZ.fits (h_out * w_out) /\
    SZ.fits (cout * h_out * w_out) /\
    SZ.fits (h_out + ph) /\
    SZ.fits (w_out + pw) /\
    SZ.fits (kh * dh) /\
    SZ.fits (kw * dw) /\
    b * cout * h_out * w_out <= max_blocks * max_threads

inline_for_extraction noextract
val convt2d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (sh sw : szp) (ph pw : sz) (dh dw : szp)
  (h_out w_out : szp)
  (#lx : layout1 (b * cin * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (cin * cout * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*cin*h_in*w_in)))
  (#sw_l : erased (chest1 et (cin*cout*kh*kw)))
  (#sbias : erased (chest1 et cout))
  (#sy0 : erased (chest1 et (b*cout*h_out*w_out)))
  (#fx #fw #fb : perm)
  : stt unit
    (requires
      cpu **
      on gpu_loc (gx |-> Frac fx sx) **
      on gpu_loc (gw |-> Frac fw sw_l) **
      on gpu_loc (gbias |-> Frac fb sbias) **
      on gpu_loc (gy |-> sy0) **
      pure (is_global gx /\ is_global gw /\
            is_global gbias /\ is_global gy /\
            convT2d_size_req b cin h_in w_in cout kh kw
                             sh sw ph pw dh dw h_out w_out))
    (ensures fun _ ->
      cpu **
      on gpu_loc (gx |-> Frac fx sx) **
      on gpu_loc (gw |-> Frac fw sw_l) **
      on gpu_loc (gbias |-> Frac fb sbias) **
      (exists* (sy : chest1 et (b*cout*h_out*w_out)).
        on gpu_loc (gy |-> sy) **
        pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
                acc1 sy tid ==
                convT2d_out_at b cin h_in w_in cout kh kw
                               sh sw ph pw dh dw
                               h_out w_out sx sw_l sbias tid)))
