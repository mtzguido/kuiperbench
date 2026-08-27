module Kuiper.Kernel.Conv2D.Depthwise

(* Implementation of [Kuiper.Kernel.Conv2D.Depthwise].

   See the [.fsti] for the contract and the spec correspondence.  The
   kernel computes one output pixel per thread; the inner accumulation
   over the [(kh_i, kw_i)] taps is a single while-loop matched up
   to [Kuiper.Spec.DepthwiseConv2D.__dwconv2d_single].

   Mirrors [Kuiper.Kernel.Conv2D.Naive] minus the input-channel axis.
   Setup, teardown, and kpre/kpost sendability are all discharged at
   the [kdesc] level.  The per-thread loop's spec
   connection to [Kuiper.Spec.DepthwiseConv2D.__dwconv2d_single] is
   discharged via the [dwconv2d_partial_at] / [dwconv2d_partial_at_step]
   pattern (mirrors [Kuiper.Kernel.Conv1D.Naive]). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv2D
open Kuiper.Spec.DepthwiseConv2D
open Kuiper.Kernel.Conv2D.Naive
open Kuiper.Bijection { ( =~ ) }
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

(* [abs (n @| INil)] is definitionally [natlt n & unit]; expose this to the SMT
   solver so that the abstract 1-D tensor index unifies with the explicit
   [(i, ())] tuples produced by reads/writes and [forevery] reindexings. *)
let abs_cons_nil_eq (n:nat)
  : Lemma (Kuiper.Shape.abs (n @| INil) == (natlt n & unit))
          [SMTPat (Kuiper.Shape.abs (n @| INil))]
  = ()

unfold
let abs_bij (#len : nat) : (Kuiper.Shape.abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }


(* Per-thread pre/post predicates. *)

unfold
let kpre
  (#et:Type) {| scalar et |}
  (b c h_in w_in : pos) (kh kw : pos)
  (h_out w_out : pos)
  (#lx : layout1 (b * c * h_in * w_in))
  (#lw : layout1 (c * 1 * kh * kw))
  (#lbias : layout1 c)
  (#ly : layout1 (b * c * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*c*h_in*w_in))
  (sw : chest1 et (c*1*kh*kw))
  (sbias : chest1 et c)
  (sy0 : chest1 et (b*c*h_out*w_out))
  (fx fw fb : perm)
  (tid : natlt (b * c * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * c * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * c * h_out * w_out)) sw **
    gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> acc1 sy0 tid

unfold
let kpost
  (#et:Type) {| scalar et |}
  (b c h_in w_in : pos) (kh kw : pos)
  (stride : pos) (pad : nat)
  (h_out w_out : pos)
  (#lx : layout1 (b * c * h_in * w_in))
  (#lw : layout1 (c * 1 * kh * kw))
  (#lbias : layout1 c)
  (#ly : layout1 (b * c * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*c*h_in*w_in))
  (sw : chest1 et (c*1*kh*kw))
  (sbias : chest1 et c)
  (fx fw fb : perm)
  (tid : natlt (b * c * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * c * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * c * h_out * w_out)) sw **
    gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> dwconv2d_out_at b c h_in w_in kh kw stride pad
                                    h_out w_out sx sw sbias tid

#push-options "--z3rlimit 60"

(* Inner-loop helper: read a tap from x at [(bi, ci, h_idx, w_idx)],
   with zero-padded out-of-range guards. *)
inline_for_extraction noextract
fn read_x_padded
  (#et : Type0) {| scalar et |}
  (b c h_in w_in : szp)
  (#lx : layout1 (b * c * h_in * w_in)) {| ctlayout lx |}
  (gx : array1 et lx)
  (#sx : chest1 et (b*c*h_in*w_in))
  (#fx : perm)
  (bi : szlt b)
  (ci : szlt c)
  (h_signed : sz)
  (w_signed : sz)
  (pad : sz)
  (#_ : squash (SZ.fits (b * c * h_in * w_in)))
  preserves
    gpu **
    gx |-> Frac fx sx
  returns
    v : et
  ensures
    pure (
      let h_int : int = SZ.v h_signed - SZ.v pad in
      let w_int : int = SZ.v w_signed - SZ.v pad in
      v == (if 0 <= h_int && h_int < h_in &&
               0 <= w_int && w_int < w_in
            then acc1 sx (((bi * c + ci) * h_in + h_int) * w_in + w_int)
            else zero))
{
  if (pad <=^ h_signed && pad <=^ w_signed) {
    let hi = h_signed -^ pad;
    let wi = w_signed -^ pad;
    if (hi <^ h_in && wi <^ w_in) {
      let bc = bi *^ c;
      let bc_ci = bc +^ ci;
      let bc_ci_h = bc_ci *^ h_in;
      let bc_ci_h_hi = bc_ci_h +^ hi;
      let bc_ci_h_hi_w = bc_ci_h_hi *^ w_in;
      let flat : szlt (b * c * h_in * w_in) = bc_ci_h_hi_w +^ wi;
      let v = tensor_read gx (flat, ());
      v
    } else {
      zero
    }
  } else {
    zero
  }
}

#pop-options

#push-options "--z3rlimit 60"

(* Read a depthwise-weight tap [(ci, 0, kh_i, kw_i)] from the flat weight
   array of shape (c, 1, kh, kw). *)
inline_for_extraction noextract
fn read_w_tap
  (#et : Type0) {| scalar et |}
  (c kh kw : szp)
  (#lw : layout1 (c * 1 * kh * kw)) {| ctlayout lw |}
  (gw : array1 et lw)
  (#sw : chest1 et (c*1*kh*kw))
  (#fw : perm)
  (ci : szlt c)
  (kh_i : szlt kh) (kw_i : szlt kw)
  (#_ : squash (SZ.fits (c * 1 * kh * kw)))
  preserves
    gpu **
    gw |-> Frac fw sw
  returns
    v : et
  ensures
    pure (v == acc1 sw (((ci * 1 + 0) * kh + kh_i) * kw + kw_i))
{
  Math.lemma_mult_le_right kh (ci + 1) c;
  Math.lemma_mult_le_right kw (ci * kh + kh_i + 1) (c * kh);
  let p1 = ci *^ kh +^ kh_i;
  let flat : szlt (c * 1 * kh * kw) = p1 *^ kw +^ kw_i;
  tensor_read gw (flat, ())
}

#pop-options

#push-options "--z3rlimit 200 --fuel 2 --ifuel 1"

(* Local helper: partial depthwise sum over the linearised (kh_i, kw_i)
   index up to [to], with all parameters explicit.  Used as the loop-
   invariant predicate for [kf]'s accumulator.  This kernel uses
   stride / pad with dilation = 1 for both axes, so the partial-sum
   wrapper specialises [__dwconv2d_single] at [d_h = d_w = 1] and
   [s_h = s_w = stride], [p_h = p_w = pad]. *)
unfold
let dwconv2d_partial_at
  (#et : Type) {| scalar et |}
  (b c h_in w_in : pos)
  (kh kw : pos) (stride : pos) (pad : nat)
  (h_out w_out : nat)
  (sx : chest1 et (b*c*h_in*w_in))
  (sw : chest1 et (c*1*kh*kw))
  (bi : natlt b) (ci : natlt c)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= kh * kw})
  : GTot et
  = __dwconv2d_single kh kw stride stride pad pad 1 1
      (lseq_to_t4 b c h_in w_in sx)
      (lseq_to_t4 c 1 kh kw sw)
      bi ci oh ow to

(* Step lemma for [dwconv2d_partial_at]: extends the partial sum by one tap. *)
let dwconv2d_partial_at_step
  (#et : Type) {| scalar et |}
  (b c h_in w_in : pos)
  (kh kw : pos) (stride : pos) (pad : nat)
  (h_out w_out : nat)
  (sx : chest1 et (b*c*h_in*w_in))
  (sw : chest1 et (c*1*kh*kw))
  (bi : natlt b) (ci : natlt c)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let kh_i = unrank_dw_kh kh kw i in
      let kw_i = unrank_dw_kw kh kw i in
      let h_idx : int = oh * stride + kh_i - pad in
      let w_idx : int = ow * stride + kw_i - pad in
      dwconv2d_partial_at b c h_in w_in kh kw stride pad h_out w_out
                          sx sw bi ci oh ow to ==
      add (dwconv2d_partial_at b c h_in w_in kh kw stride pad h_out w_out
                               sx sw bi ci oh ow (to - 1))
          (mul (read_padded (lseq_to_t4 b c h_in w_in sx) bi ci h_idx w_idx)
               (tacc (lseq_to_t4 c 1 kh kw sw) ci 0 kh_i kw_i))))
  = __dwconv2d_single_lemma kh kw stride stride pad pad 1 1
      (lseq_to_t4 b c h_in w_in sx)
      (lseq_to_t4 c 1 kh kw sw)
      bi ci oh ow to

(* Per-thread depthwise body. *)
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (b c h_in w_in : szp)
  (kh kw : szp)
  (stride : szp) (pad : sz)
  (h_out w_out : szp)
  (#lx : layout1 (b * c * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (c * 1 * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 c) {| ctlayout lbias |}
  (#ly : layout1 (b * c * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*c*h_in*w_in))
  (#sw : chest1 et (c*1*kh*kw))
  (#sbias : chest1 et c)
  (#sy0 : chest1 et (b*c*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (b * c * h_out * w_out > 0))
  (#_ : squash (SZ.fits (kh * kw) /\
                SZ.fits (h_out * w_out) /\
                SZ.fits (c * h_out * w_out) /\
                SZ.fits (b * c * h_in * w_in) /\
                SZ.fits (c * 1 * kh * kw) /\
                SZ.fits (h_out * stride + kh) /\
                SZ.fits (w_out * stride + kw)))
  (tid : szlt (b * c * h_out * w_out))
  ()
  norewrite
  requires
    gpu **
    kpre #et b c h_in w_in kh kw h_out w_out #lx #lw #lbias #ly
         gx gw gbias gy sx sw sbias sy0 fx fw fb tid
  ensures
    gpu **
    kpost #et b c h_in w_in kh kw stride pad h_out w_out #lx #lw #lbias #ly
          gx gw gbias gy sx sw sbias fx fw fb tid
{
  (* The per-thread body proves [result == dwconv2d_out_at ...] via a
     loop invariant tracking [acc == dwconv2d_partial_at ... (SZ.v k)]
     and the step lemma [dwconv2d_partial_at_step] (which wraps the
     spec-level [__dwconv2d_single_lemma]).  Setup, teardown, and
     sendability are discharged at the [kdesc] level (see below). *)
  let how : sz = h_out *^ w_out;
  let chow : sz = c *^ how;
  let bi : szlt b = tid /^ chow;
  let r1 : szlt chow = tid %^ chow;
  let ci : szlt c = r1 /^ how;
  let r2 : szlt how = r1 %^ how;
  let oh : szlt h_out = r2 /^ w_out;
  let ow : szlt w_out = r2 %^ w_out;

  let n_taps : sz = kh *^ kw;

  let oh_s : sz = oh *^ stride;
  let ow_s : sz = ow *^ stride;

  let mut acc : et = zero;
  let mut k : sz = 0sz;

  while (!k <^ n_taps)
    invariant
      exists* (vk : sz{SZ.v vk <= kh * kw}).
        k |-> vk **
        acc |-> dwconv2d_partial_at b c h_in w_in kh kw stride pad
                  h_out w_out sx sw bi ci oh ow (SZ.v vk)
    invariant gx |-> Frac (fx /. (b * c * h_out * w_out)) sx
    invariant gw |-> Frac (fw /. (b * c * h_out * w_out)) sw
    invariant gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias
    invariant gpu
    decreases (kh * kw - SZ.v !k)
  {
    let kk_v = !k;
    let kh_i : szlt kh = kk_v /^ kw;
    let kw_i : szlt kw = kk_v %^ kw;

    let h_signed = oh_s +^ kh_i;
    let w_signed = ow_s +^ kw_i;
    let xv = read_x_padded b c h_in w_in gx bi ci h_signed w_signed pad;
    let wv = read_w_tap c kh kw gw ci kh_i kw_i;
    let prod = mul xv wv;
    let acc0 = !acc;
    (* Establish the step equation: prod equals the lemma's per-tap product. *)
    assert pure (xv ==
      read_padded (lseq_to_t4 b c h_in w_in sx) bi ci
        (oh * stride + kh_i - pad) (ow * stride + kw_i - pad));
    assert pure (wv == tacc (lseq_to_t4 c 1 kh kw sw) ci 0 kh_i kw_i);
    assert pure (SZ.v kh_i == unrank_dw_kh kh kw (SZ.v kk_v));
    assert pure (SZ.v kw_i == unrank_dw_kw kh kw (SZ.v kk_v));
    dwconv2d_partial_at_step b c h_in w_in kh kw stride pad
      h_out w_out sx sw bi ci oh ow (SZ.v kk_v + 1);
    acc := add acc0 prod;
    k := !k +^ 1sz;
  };

  let bias_v = tensor_read gbias (ci, ());
  let result = add bias_v !acc;
  tensor_write_cell gy (tid, ()) result
}

#pop-options

#push-options "--z3rlimit 60"

(* Ghost setup: factor the launcher's full-permission frame into N per-thread
   slices.  Mirrors [Kuiper.Kernel.Conv2D.Naive.conv2d_naive_setup], adapted
   for the depthwise weight shape [(c, 1, kh, kw)] and N = b*c*h_out*w_out. *)
ghost
fn dwconv2d_setup
  (#et : Type0) {| scalar et |}
  (b c h_in w_in : szp)
  (kh kw : szp)
  (stride : szp)
  (h_out w_out : szp)
  (#lx : layout1 (b * c * h_in * w_in))
  (#lw : layout1 (c * 1 * kh * kw))
  (#lbias : layout1 c)
  (#ly : layout1 (b * c * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*c*h_in*w_in))
  (#sw : chest1 et (c*1*kh*kw))
  (#sbias : chest1 et c)
  (#sy0 : chest1 et (b*c*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out))
  ()
  norewrite
  requires
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw) **
    (gbias |-> Frac fb sbias) **
    (gy |-> sy0)
  ensures
    (forall+ (tid : natlt (b *^ c *^ h_out *^ w_out)).
       kpre #et b c h_in w_in kh kw h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw sbias sy0 fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
{
  tensor_pts_to_ref gy;
  tensor_share_n gx (b * c * h_out * w_out);
  tensor_share_n gw (b * c * h_out * w_out);
  tensor_share_n gbias (b * c * h_out * w_out);
  tensor_explode gy;
  forevery_iso (abs_bij #(b * c * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * c * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc sy0 i);
  forevery_ext
    (fun (i : natlt (b * c * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc sy0 (abs_bij.gg i))
    (fun (i : natlt (b * c * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * c * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias)
    (fun (i : natlt (b * c * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * c * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * c * h_out * w_out)) sw)
    (fun (i : natlt (b * c * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_zip
    (fun (_ : natlt (b * c * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * c * h_out * w_out)) sx)
    (fun (i : natlt (b * c * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * c * h_out * w_out)) sw) **
       (gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (b * c * h_out * w_out)
                   (SZ.v (b *^ c *^ h_out *^ w_out));
  ()
}

(* Ghost teardown: gather N per-thread slices back into the launcher
   postcondition.  Symmetric inverse of [dwconv2d_setup]. *)
ghost
fn dwconv2d_teardown
  (#et : Type0) {| scalar et |}
  (b c h_in w_in : szp)
  (kh kw : szp)
  (stride : szp) (pad : sz)
  (h_out w_out : szp)
  (#lx : layout1 (b * c * h_in * w_in))
  (#lw : layout1 (c * 1 * kh * kw))
  (#lbias : layout1 c)
  (#ly : layout1 (b * c * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*c*h_in*w_in))
  (#sw : chest1 et (c*1*kh*kw))
  (#sbias : chest1 et c)
  (#fx #fw #fb : perm)
  (#_ : squash (dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out))
  ()
  norewrite
  requires
    (forall+ (tid : natlt (b *^ c *^ h_out *^ w_out)).
       kpost #et b c h_in w_in kh kw stride pad h_out w_out
             #lx #lw #lbias #ly
             gx gw gbias gy sx sw sbias fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*c*h_out*w_out)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < b*c*h_out*w_out}).
               acc1 sy tid ==
               dwconv2d_out_at b c h_in w_in kh kw stride pad
                               h_out w_out sx sw sbias tid))
{
  forevery_rw_size (SZ.v (b *^ c *^ h_out *^ w_out))
                   (b * c * h_out * w_out)
    #(kpost #et b c h_in w_in kh kw stride pad h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw sbias fx fw fb);
  forevery_unzip
    (fun (_ : natlt (b * c * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * c * h_out * w_out)) sx)
    (fun (i : natlt (b * c * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * c * h_out * w_out)) sw) **
       (gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> dwconv2d_out_at b c h_in w_in kh kw stride pad
                                      h_out w_out sx sw sbias i));
  forevery_unzip
    (fun (_ : natlt (b * c * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * c * h_out * w_out)) sw)
    (fun (i : natlt (b * c * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> dwconv2d_out_at b c h_in w_in kh kw stride pad
                                      h_out w_out sx sw sbias i));
  forevery_unzip
    (fun (_ : natlt (b * c * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * c * h_out * w_out)) sbias)
    (fun (i : natlt (b * c * h_out * w_out)) ->
       Cell gy (idx1 i) |-> dwconv2d_out_at b c h_in w_in kh kw stride pad
                                     h_out w_out sx sw sbias i);
  tensor_gather_n gx (b * c * h_out * w_out);
  tensor_gather_n gw (b * c * h_out * w_out);
  tensor_gather_n gbias (b * c * h_out * w_out);
  let sy : chest1 et (b * c * h_out * w_out) =
    hide (mk1
            (fun (tid : nat{tid < b * c * h_out * w_out}) ->
               dwconv2d_out_at b c h_in w_in kh kw stride pad
                               h_out w_out sx sw sbias tid));
  forevery_ext
    (fun (i : natlt (b * c * h_out * w_out)) ->
       Cell gy (idx1 i) |-> dwconv2d_out_at b c h_in w_in kh kw stride pad
                                     h_out w_out sx sw sbias i)
    (fun (i : natlt (b * c * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc (reveal sy) (abs_bij.gg i));
  forevery_iso_back (abs_bij #(b * c * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * c * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc (reveal sy) i);
  tensor_implode gy;
  ()
}

#pop-options

#push-options "--z3rlimit 200 --fuel 2 --ifuel 2"

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (b c h_in w_in : szp)
  (kh kw : szp)
  (stride : szp) (pad : sz)
  (h_out w_out : szp)
  (#lx : layout1 (b * c * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (c * 1 * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 c) {| ctlayout lbias |}
  (#ly : layout1 (b * c * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*c*h_in*w_in))
  (#sw : chest1 et (c*1*kh*kw))
  (#sbias : chest1 et c)
  (#sy0 : chest1 et (b*c*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (is_global gx /\ is_global gw /\
                is_global gbias /\ is_global gy /\
                dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out))
  : kernel_desc
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 et (b*c*h_out*w_out)).
         (gy |-> sy) **
         pure (forall (tid : nat{tid < b*c*h_out*w_out}).
                 acc1 sy tid ==
                 dwconv2d_out_at b c h_in w_in kh kw stride pad
                                 h_out w_out sx sw sbias tid)))
=
{
  nthr = b *^ c *^ h_out *^ w_out;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = dwconv2d_setup b c h_in w_in kh kw stride h_out w_out
                            gx gw gbias gy;
  teardown = dwconv2d_teardown b c h_in w_in kh kw stride pad h_out w_out
                               gx gw gbias gy;
  kpre  = kpre #et b c h_in w_in kh kw h_out w_out #lx #lw #lbias #ly gx gw gbias gy sx sw sbias sy0 fx fw fb;
  kpost = kpost #et b c h_in w_in kh kw stride pad h_out w_out #lx #lw #lbias #ly gx gw gbias gy sx sw sbias fx fw fb;
  f = kf b c h_in w_in kh kw stride pad h_out w_out gx gw gbias gy;
  (* Inherited tree-wide debt (sendability of compound per-thread slprops). *)
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn dwconv2d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b c h_in w_in : szp)
  (kh kw : szp)
  (stride : szp) (pad : sz)
  (h_out w_out : szp)
  (#lx : layout1 (b * c * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (c * 1 * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 c) {| ctlayout lbias |}
  (#ly : layout1 (b * c * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*c*h_in*w_in))
  (#sw : chest1 et (c*1*kh*kw))
  (#sbias : chest1 et c)
  (#sy0 : chest1 et (b*c*h_out*w_out))
  (#fx #fw #fb : perm)
  norewrite
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
{
  launch_sync (kdesc b c h_in w_in kh kw stride pad h_out w_out gx gw gbias gy)
}
