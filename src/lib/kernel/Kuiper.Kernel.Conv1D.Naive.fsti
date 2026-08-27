module Kuiper.Kernel.Conv1D.Naive

(* Naive 1D convolution forward kernel: one thread per output element.

   Operates on flat NCL row-major array1 buffers:
     x      : array1 et (b*cin*l_in)
     weight : array1 et (cout*cin*kk)
     bias   : array1 et cout
     y      : array1 et (b*cout*l_out)

   Computes
     y[b, oc, ol] = bias[oc]
        + Σ_{ic, k} x[b, ic, ol*stride + k*dilation - pad]
                    * weight[oc, ic, k]
   with zero-padded out-of-range reads, groups = 1.

   The post-condition ties output cells to [Kuiper.Spec.Conv1D.conv1d_single]
   via the canonical row-major flattening [lseq_to_t3]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv1D
module Seq = FStar.Seq
module SZ = Kuiper.SizeT

(* Row-major flattening of an lseq into an [etensor3]. *)
[@@erasable]
val lseq_to_t3
  (#et:Type) (d0 d1 d2 : nat)
  (s : chest1 et (d0 * d1 * d2))
  : etensor3 et d0 d1 d2

val lseq_to_t3_index
  (#et:Type) (d0 d1 d2 : nat)
  (s : chest1 et (d0 * d1 * d2))
  (i:natlt d0) (j:natlt d1) (k:natlt d2)
  : Lemma (t3acc (lseq_to_t3 d0 d1 d2 s) i j k ==
           acc1 s ((i * d1 + j) * d2 + k))
          [SMTPat (t3acc (lseq_to_t3 d0 d1 d2 s) i j k)]

(* Per-thread post-condition predicate (over output index [tid]).  Decodes
   [tid] into (b, oc, ol) via row-major unflatten, then evaluates
   [conv1d_single] on the etensor3 induced by the flat input/weight lseqs. *)
let conv1d_out_at
  (#et:Type) {| scalar et |}
  (b cin l_in : nat)
  (cout : nat) (kk : pos)
  (stride : pos) (pad : nat) (dilation : pos)
  (l_out : nat)
  (sx : chest1 et (b*cin*l_in))
  (sw : chest1 et (cout*cin*kk))
  (sbias : chest1 et cout)
  (tid : nat{tid < b*cout*l_out})
  : GTot et
  = let bi : natlt b = tid / (cout*l_out) in
    let r1 = tid % (cout*l_out) in
    let oc : natlt cout = r1 / l_out in
    let ol : natlt l_out = r1 % l_out in
    conv1d_single kk stride pad dilation
      (lseq_to_t3 b cin l_in sx)
      (lseq_to_t3 cout cin kk sw)
      (chest1_to_seq sbias) bi oc ol

(* SizeT precondition we require from callers. *)
unfold
let conv1d_size_req
  (b cin l_in cout kk : nat)
  (stride : nat) (dilation : nat)
  (l_out : nat)
  : prop
  = SZ.fits (b * cin * l_in) /\
    SZ.fits (cout * cin * kk) /\
    SZ.fits (b * cout * l_out) /\
    SZ.fits (cin * kk) /\
    SZ.fits (cout * l_out) /\
    SZ.fits (l_out * stride + kk * dilation) /\
    b * cout * l_out <= max_blocks * max_threads

inline_for_extraction noextract
fn conv1d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b cin l_in cout kk stride : szp)
  (pad : sz)
  (dilation l_out : szp)
  (#lx : layout1 (b * cin * l_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kk)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * l_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*cin*l_in)))
  (#sw : erased (chest1 et (cout*cin*kk)))
  (#sbias : erased (chest1 et cout))
  (#sy0 : erased (chest1 et (b*cout*l_out)))
  (#fx #fw #fb : perm)
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          conv1d_size_req b cin l_in cout kk stride dilation l_out)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*l_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*l_out}).
               acc1 sy tid ==
               conv1d_out_at b cin l_in cout kk stride pad dilation
                             l_out sx sw sbias tid))
