module Kuiper.KB.SeparableConv2D

(* KernelBench L1 #86 — depthwise-separable 2D convolution forward.

   This is the COMPOSED entry point that ties the two-call chain
   [pointwise(depthwise(x))] to the *whole* separable-conv functional
   spec [Kuiper.Spec.SeparableConv2D.separable_conv2d], rather than
   merely chaining two independent posts.

   The boundary self-allocates both the intermediate (depthwise output)
   scratch buffer and the final output buffer:

     1. allocate [mid : B*C*H_out*W_out], run the verified depthwise
        kernel [Kuiper.KB.DepthwiseConv2D.dwconv2d_f32] into it;
     2. allocate [y : B*C_out*H_out*W_out], run the verified general
        conv kernel [Kuiper.KB.Conv2DGeneral.conv2d_general_f32] with
        kh=kw=1, stride=1, pad=0 (a 1x1 / pointwise conv) reading [mid];
     3. free [mid], return [y].

   The post proves each output cell equals [separable_out_at] — the
   per-pixel evaluation of [separable_conv2d] applied to the ORIGINAL
   inputs (depthwise weight/bias and pointwise weight/bias).  The pure
   composition proof lives in the .fst and rests on
   [Kuiper.Spec.SeparableConv2D.lemma_separable_conv2d_index] plus a
   1x1-conv = pointwise equivalence and a mixed-radix index round-trip,
   all ADMIT-free. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Spec.DepthwiseConv2D
open Kuiper.Spec.PointwiseConv2D
open Kuiper.Spec.SeparableConv2D
open Kuiper.Kernel.Conv2D.Naive
open Kuiper.Kernel.Conv2D.Depthwise
module Seq = FStar.Seq
module SZ = Kuiper.SizeT

(* Reinterpret the flat pointwise weight [(C_out, C, 1, 1)] as the
   [(C_out, C)] erased matrix the pointwise spec consumes.  Both index
   the same flat seq at [oc*c + ic]. *)
unfold
let lseq_to_mat
  (#et:Type) {| scalar et |}
  (cout c : nat)
  (s : chest1 et (cout * c * 1 * 1))
  : chest2 et cout c
  = mk2 (fun oc ic -> tacc (lseq_to_t4 cout c 1 1 s) oc ic 0 0)

(* Per-thread post-condition predicate for the composed separable conv.
   Decodes [tid] into (b, oc, oh, ow) by row-major unflatten over
   [(B, C_out, H_out, W_out)], then evaluates the *whole* separable
   conv spec at that pixel. *)
let separable_out_at
  (#et:Type) {| scalar et |}
  (b c h_in w_in : nat)
  (kh : pos) (kw : pos)
  (stride : pos) (pad : nat)
  (cout : nat)
  (h_out w_out : nat)
  (sx : chest1 et (b*c*h_in*w_in))
  (sw_dw : chest1 et (c*1*kh*kw))
  (sbias_dw : chest1 et c)
  (sw_pw : chest1 et (cout*c*1*1))
  (sbias_pw : chest1 et cout)
  (tid : nat{tid < b*cout*h_out*w_out})
  : GTot et
  = let bi : natlt b = tid / (cout*h_out*w_out) in
    let r1 = tid % (cout*h_out*w_out) in
    let oc : natlt cout = r1 / (h_out*w_out) in
    let r2 = r1 % (h_out*w_out) in
    let oh : natlt h_out = r2 / w_out in
    let ow : natlt w_out = r2 % w_out in
    tacc (separable_conv2d kh kw stride stride pad pad 1 1 h_out w_out
            (lseq_to_t4 b c h_in w_in sx)
            (lseq_to_t4 c 1 kh kw sw_dw)
            (chest1_to_seq sbias_dw)
            (lseq_to_mat cout c sw_pw)
            (chest1_to_seq sbias_pw))
         bi oc oh ow

(* Verified output spatial dimension (re-exported from the depthwise
   stage, which fixes the separable-conv output size).  The bridge uses
   this instead of a hand-rolled C++ division. *)
val separable_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)

(* Combined size_t precondition: depthwise stage + 1x1 pointwise stage. *)
unfold
let separable_size_req
  (b c h_in w_in kh kw : nat)
  (stride : nat)
  (cout h_out w_out : nat)
  : prop
  = dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out /\
    conv2d_size_req b c h_out w_out cout 1 1 1 h_out w_out

inline_for_extraction noextract
type separable_alloc_ty =
  fn
  (b c h_in w_in kh kw stride : szp)
  (pad : sz)
  (cout h_out : szp)
  (w_out : szp { separable_size_req b c h_in w_in kh kw stride cout h_out w_out })
  (gx : array1 f32 (l1_forward (b * c * h_in * w_in))
        { is_global gx })
  (gw_dw : array1 f32 (l1_forward (c * 1 * kh * kw))
        { is_global gw_dw })
  (gbias_dw : array1 f32 (l1_forward c)
        { is_global gbias_dw })
  (gw_pw : array1 f32 (l1_forward (cout * c * 1 * 1))
        { is_global gw_pw })
  (gbias_pw : array1 f32 (l1_forward cout)
        { is_global gbias_pw })
  (#fx #fwd #fbd #fwp #fbp : perm)
  (#sx : chest1 f32 (b * c * h_in * w_in))
  (#sw_dw : chest1 f32 (c * 1 * kh * kw))
  (#sbias_dw : chest1 f32 c)
  (#sw_pw : chest1 f32 (cout * c * 1 * 1))
  (#sbias_pw : chest1 f32 cout)
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw_dw |-> Frac fwd sw_dw) **
    on gpu_loc (gbias_dw |-> Frac fbd sbias_dw) **
    on gpu_loc (gw_pw |-> Frac fwp sw_pw) **
    on gpu_loc (gbias_pw |-> Frac fbp sbias_pw)
  returns gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw_dw |-> Frac fwd sw_dw) **
    on gpu_loc (gbias_dw |-> Frac fbd sbias_dw) **
    on gpu_loc (gw_pw |-> Frac fwp sw_pw) **
    on gpu_loc (gbias_pw |-> Frac fbp sbias_pw) **
    (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
               acc1 sy tid ==
               separable_out_at b c h_in w_in kh kw stride pad
                                cout h_out w_out
                                sx sw_dw sbias_dw sw_pw sbias_pw tid))

val separable_alloc_f32 : separable_alloc_ty

inline_for_extraction let () = ()
