module Kuiper.Kernel.Conv3D.Naive

(* Implementation of [Kuiper.Kernel.Conv3D.Naive].

   See the [.fsti] for the contract and the spec correspondence.  The
   kernel computes one output voxel per thread; the inner accumulation
   over the [(ic, kd_i, kh_i, kw_i)] taps is a single while-loop matched
   up to [Kuiper.Spec.Conv3D.__conv3d_single] (with dilation = 1).

   Setup, teardown, and kpre/kpost sendability are all discharged at
   the [kdesc] level.
   The spec-connection [assume pure] for [conv3d_out_at] has been
   discharged using the [conv3d_partial_at] / [conv3d_partial_at_step]
   pattern from [Kuiper.Kernel.Conv1D.Naive]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv3D
open FStar.FunctionalExtensionality { (^->>) }
open Kuiper.Bijection { ( =~ ) }
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

let div_lt_product (x b : nat) (d : pos)
  : Lemma (requires x < b * d) (ensures x / d < b)
  = let q = x / d in
    Math.division_definition x d q;
    ()

let output_decode_facts
  (b cout d h w : pos)
  (tid : natlt (b * cout * d * h * w))
  (how dhw cdhw : nat)
  : Lemma
      (requires
        how == h * w /\
        dhw == d * how /\
        cdhw == cout * dhw)
      (ensures
        cdhw == cout * d * h * w /\
        tid < b * cdhw)
  = ()

let named_mul_value
  (x : sz)
  (y : sz{FStar.SizeT.fits (SZ.v x * SZ.v y)})
  (xy : sz)
  : Lemma
      (requires xy == SZ.mul x y)
      (ensures SZ.v xy == SZ.v x * SZ.v y)
  = ()

let flatten_taps
  (cin kd kh kw kh_kw kd_kh_kw n_taps : nat)
  : Lemma
      (requires
        kh_kw == kh * kw /\
        kd_kh_kw == kd * kh_kw /\
        n_taps == cin * kd_kh_kw)
      (ensures n_taps == cin * kd * kh * kw)
  = ()

let flattened_taps_fit
  (cin kd kh kw kh_kw kd_kh_kw : nat)
  : Lemma
      (requires
        SZ.fits (cin * kd * kh * kw) /\
        kh_kw == kh * kw /\
        kd_kh_kw == kd * kh_kw)
      (ensures SZ.fits (cin * kd_kh_kw))
  = ()

let unrank3_from_steps
  (cin kd kh kw : pos)
  (i : nat { i < cin * kd * kh * kw })
  (kh_kw kd_kh_kw n_taps ic r kd_i r2 kh_i kw_i : nat)
  : Lemma
      (requires
        kh_kw == kh * kw /\ kd_kh_kw == kd * kh_kw /\
        n_taps == cin * kd_kh_kw /\
        ic == i / kd_kh_kw /\ r == i % kd_kh_kw /\
        kd_i == r / kh_kw /\ r2 == r % kh_kw /\
        kh_i == r2 / kw /\ kw_i == r2 % kw)
      (ensures
        ic == unrank3_ic cin kd kh kw i /\
        kd_i == unrank3_kd cin kd kh kw i /\
        kh_i == unrank3_kh cin kd kh kw i /\
        kw_i == unrank3_kw cin kd kh kw i)
  = ()

let decreases_after_increment (bound k : nat)
  : Lemma (requires k < bound) (ensures (bound - (k + 1) < bound - k))
  = ()

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


let lseq_to_t5
  (#et:Type) (d0 d1 d2 d3 d4 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3 * d4))
  : etensor5 et d0 d1 d2 d3 d4
  = mkT5 (fun i j k l m ->
            acc1 s ((((i * d1 + j) * d2 + k) * d3 + l) * d4 + m))

let lseq_to_t5_index
  (#et:Type) (d0 d1 d2 d3 d4 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3 * d4))
  (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3) (m:natlt d4)
  : Lemma (t5acc (lseq_to_t5 d0 d1 d2 d3 d4 s) i j k l m ==
           acc1 s ((((i * d1 + j) * d2 + k) * d3 + l) * d4 + m))
          [SMTPat (t5acc (lseq_to_t5 d0 d1 d2 d3 d4 s) i j k l m)]
  = ()

(* Per-thread pre/post predicates. *)

unfold
let kpre
  (#et:Type) {| scalar et |}
  (b cin d_in h_in w_in cout : pos) (kd kh kw : pos)
  (d_out h_out w_out : pos)
  (#lx : layout1 (b * cin * d_in * h_in * w_in))
  (#lw : layout1 (cout * cin * kd * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * d_out * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*cin*d_in*h_in*w_in))
  (sw : chest1 et (cout*cin*kd*kh*kw))
  (sbias : chest1 et cout)
  (sy0 : chest1 et (b*cout*d_out*h_out*w_out))
  (fx fw fb : perm)
  (tid : natlt (b * cout * d_out * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * d_out * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * cout * d_out * h_out * w_out)) sw **
    gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> acc1 sy0 tid

unfold
let kpost
  (#et:Type) {| scalar et |}
  (b cin d_in h_in w_in cout : pos) (kd kh kw : pos)
  (stride : pos) (pad : nat)
  (d_out h_out w_out : pos)
  (#lx : layout1 (b * cin * d_in * h_in * w_in))
  (#lw : layout1 (cout * cin * kd * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * d_out * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*cin*d_in*h_in*w_in))
  (sw : chest1 et (cout*cin*kd*kh*kw))
  (sbias : chest1 et cout)
  (fx fw fb : perm)
  (tid : natlt (b * cout * d_out * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * d_out * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * cout * d_out * h_out * w_out)) sw **
    gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> conv3d_out_at b cin d_in h_in w_in cout kd kh kw
                                  stride pad d_out h_out w_out
                                  sx sw sbias tid

#push-options "--z3rlimit 50"

(* Inner-loop helper: read a tap from x at (bi, ic, d_signed-pad, h_signed-pad,
   w_signed-pad) with zero-padded out-of-range guards. *)
inline_for_extraction noextract
fn read_x_padded
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in : szp)
  (#lx : layout1 (b * cin * d_in * h_in * w_in)) {| ctlayout lx |}
  (gx : array1 et lx)
  (#sx : erased (chest1 et (b*cin*d_in*h_in*w_in)))
  (#fx : perm)
  (bi : szlt b)
  (ic : szlt cin)
  (d_signed : sz) (* od*stride + kd_i *)
  (h_signed : sz) (* oh*stride + kh_i *)
  (w_signed : sz) (* ow*stride + kw_i *)
  (pad : sz)
  (#_ : squash (SZ.fits (b * cin * d_in * h_in * w_in)))
  preserves
    gpu **
    gx |-> Frac fx sx
  returns
    v : et
  ensures
    pure (
      let d_int : int = SZ.v d_signed - SZ.v pad in
      let h_int : int = SZ.v h_signed - SZ.v pad in
      let w_int : int = SZ.v w_signed - SZ.v pad in
      v == (if 0 <= d_int && d_int < d_in &&
               0 <= h_int && h_int < h_in &&
               0 <= w_int && w_int < w_in
            then acc1 sx ((((bi * cin + ic) * d_in + d_int) * h_in
                                                                + h_int) * w_in + w_int)
            else zero))
{
  if (pad <=^ d_signed && pad <=^ h_signed && pad <=^ w_signed) {
    let di = d_signed -^ pad;
    let hi = h_signed -^ pad;
    let wi = w_signed -^ pad;
    if (di <^ d_in && hi <^ h_in && wi <^ w_in) {
      (* Bound [flat] for the [cit_fits] refinement of [tensor_read].  The
         same rank-bound chain as [read_w_tap]; the removed [macc_pat]
         SMTPat no longer discharges this nonlinear fact automatically. *)
      Math.lemma_mult_lt_right cin bi b;
      Math.lemma_mult_le_right d_in (bi * cin + ic + 1) (b * cin);
      Math.lemma_mult_le_right h_in ((bi * cin + ic) * d_in + di + 1)
                                    (b * cin * d_in);
      Math.lemma_mult_le_right w_in (((bi * cin + ic) * d_in + di) * h_in + hi + 1)
                                    (b * cin * d_in * h_in);
      let p1 = bi *^ cin +^ ic;
      let p2 = p1 *^ d_in +^ di;
      let p3 = p2 *^ h_in +^ hi;
      let flat : szlt (b * cin * d_in * h_in * w_in) = p3 *^ w_in +^ wi;
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

#push-options "--z3rlimit 50"

(* Read a weight tap (oc, ic, kd_i, kh_i, kw_i) from the flat weight array. *)
inline_for_extraction noextract
fn read_w_tap
  (#et : Type0) {| scalar et |}
  (cout cin kd kh kw : szp)
  (#lw : layout1 (cout * cin * kd * kh * kw)) {| ctlayout lw |}
  (gw : array1 et lw)
  (#sw : erased (chest1 et (cout*cin*kd*kh*kw)))
  (#fw : perm)
  (oc : szlt cout) (ic : szlt cin)
  (kd_i : szlt kd) (kh_i : szlt kh) (kw_i : szlt kw)
  (#_ : squash (SZ.fits (cout * cin * kd * kh * kw)))
  preserves
    gpu **
    gw |-> Frac fw sw
  returns
    v : et
  ensures
    pure (v == acc1 sw
              ((((oc * cin + ic) * kd + kd_i) * kh + kh_i) * kw + kw_i))
{
  Math.lemma_mult_lt_right cin oc cout;
  Math.lemma_mult_le_right kd (oc * cin + ic + 1) (cout * cin);
  Math.lemma_mult_le_right kh ((oc * cin + ic) * kd + kd_i + 1) (cout * cin * kd);
  Math.lemma_mult_le_right kw (((oc * cin + ic) * kd + kd_i) * kh + kh_i + 1)
                              (cout * cin * kd * kh);
  let p1 = oc *^ cin +^ ic;
  let p2 = p1 *^ kd +^ kd_i;
  let p3 = p2 *^ kh +^ kh_i;
  let flat : szlt (cout * cin * kd * kh * kw) = p3 *^ kw +^ kw_i;
  tensor_read gw (flat, ())
}

#pop-options

#push-options "--z3rlimit 50 --fuel 2 --ifuel 1"

(* Local helper: partial conv3d sum over the linearised
   [(ic, kd_i, kh_i, kw_i)] index up to [to], with all parameters
   explicit and dilation hard-wired to 1 (matching the kernel and
   [conv3d_out_at]).  Used as the loop-invariant predicate for
   [kf]'s accumulator.  Wrapping [__conv3d_single] avoids unification
   difficulties that arise from passing the [val]-only spec function
   directly inside a Pulse [exists*] slprop. *)
unfold
let conv3d_partial_at
  (#et : Type) {| scalar et |}
  (b cin d_in h_in w_in cout : pos)
  (kd kh kw : pos) (stride : pos) (pad : nat)
  (d_out h_out w_out : nat)
  (sx : chest1 et (b*cin*d_in*h_in*w_in))
  (sw : chest1 et (cout*cin*kd*kh*kw))
  (bi : natlt b) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= cin * kd * kh * kw})
  : GTot et
  = __conv3d_single kd kh kw stride stride stride pad pad pad 1 1 1
      (lseq_to_t5 b cin d_in h_in w_in sx)
      (lseq_to_t5 cout cin kd kh kw sw)
      bi oc od oh ow to

(* Step lemma for [conv3d_partial_at]: extends the partial sum by one tap. *)
let conv3d_partial_at_step
  (#et : Type) {| scalar et |}
  (b cin d_in h_in w_in cout : pos)
  (kd kh kw : pos) (stride : pos) (pad : nat)
  (d_out h_out w_out : nat)
  (sx : chest1 et (b*cin*d_in*h_in*w_in))
  (sw : chest1 et (cout*cin*kd*kh*kw))
  (bi : natlt b) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin * kd * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank3_ic cin kd kh kw i in
      let kd_i = unrank3_kd cin kd kh kw i in
      let kh_i = unrank3_kh cin kd kh kw i in
      let kw_i = unrank3_kw cin kd kh kw i in
      let d_idx : int = od * stride + kd_i * 1 - pad in
      let h_idx : int = oh * stride + kh_i * 1 - pad in
      let w_idx : int = ow * stride + kw_i * 1 - pad in
      conv3d_partial_at b cin d_in h_in w_in cout kd kh kw stride pad
                        d_out h_out w_out sx sw bi oc od oh ow to ==
      add (conv3d_partial_at b cin d_in h_in w_in cout kd kh kw stride pad
                              d_out h_out w_out sx sw bi oc od oh ow (to - 1))
          (mul (read_padded3 (lseq_to_t5 b cin d_in h_in w_in sx) bi ic
                              d_idx h_idx w_idx)
               (t5acc (lseq_to_t5 cout cin kd kh kw sw)
                      oc ic kd_i kh_i kw_i))))
  = __conv3d_single_lemma cin kd kh kw stride stride stride pad pad pad 1 1 1
      (lseq_to_t5 b cin d_in h_in w_in sx)
      (lseq_to_t5 cout cin kd kh kw sw)
      bi oc od oh ow to

#pop-options

#push-options "--z3rlimit 400 --fuel 2 --ifuel 1"

(* Per-thread conv body: decode tid, run the inner accumulator loop,
   add bias, write to output cell.  The body proves
   [result == conv3d_out_at ...] via a loop invariant tracking
   [acc == conv3d_partial_at ... (SZ.v k)] and the step lemma
   [conv3d_partial_at_step] (which wraps the spec-level
   [__conv3d_single_lemma]).  Setup, teardown, and sendability are
   discharged at the [kdesc] level (see below). *)
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (stride : szp) (pad : sz)
  (d_out h_out w_out : szp)
  (#lx : layout1 (b * cin * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*cin*d_in*h_in*w_in)))
  (#sw : erased (chest1 et (cout*cin*kd*kh*kw)))
  (#sbias : erased (chest1 et cout))
  (#sy0 : erased (chest1 et (b*cout*d_out*h_out*w_out)))
  (#fx #fw #fb : perm)
  (#_ : squash (b * cout * d_out * h_out * w_out > 0))
  (#_ : squash (SZ.fits (cin * kd * kh * kw) /\
                SZ.fits (kd * kh * kw) /\
                SZ.fits (kh * kw) /\
                SZ.fits (h_out * w_out) /\
                SZ.fits (d_out * h_out * w_out) /\
                SZ.fits (cout * d_out * h_out * w_out) /\
                SZ.fits (b * cin * d_in * h_in * w_in) /\
                SZ.fits (cout * cin * kd * kh * kw) /\
                SZ.fits (d_out * stride + kd) /\
                SZ.fits (h_out * stride + kh) /\
                SZ.fits (w_out * stride + kw)))
  (tid : szlt (b * cout * d_out * h_out * w_out))
  ()
  norewrite
  requires
    gpu **
    kpre #et b cin d_in h_in w_in cout kd kh kw d_out h_out w_out
         #lx #lw #lbias #ly gx gw gbias gy sx sw sbias sy0 fx fw fb tid
  ensures
    gpu **
    kpost #et b cin d_in h_in w_in cout kd kh kw stride pad d_out h_out w_out
          #lx #lw #lbias #ly gx gw gbias gy sx sw sbias fx fw fb tid
{
  let how : sz = h_out *^ w_out;
  named_mul_value h_out w_out how;
  let dhw : sz = d_out *^ how;
  named_mul_value d_out how dhw;
  let cdhw : sz = cout *^ dhw;
  named_mul_value cout dhw cdhw;
  output_decode_facts (SZ.v b) (SZ.v cout) (SZ.v d_out) (SZ.v h_out)
    (SZ.v w_out) (SZ.v tid) (SZ.v how) (SZ.v dhw) (SZ.v cdhw);
  div_lt_product (SZ.v tid) (SZ.v b) (SZ.v cdhw);
  let bi : szlt b = tid /^ cdhw;
  let r1 : szlt cdhw = tid %^ cdhw;
  let oc : szlt cout = r1 /^ dhw;
  let r2 : szlt dhw = r1 %^ dhw;
  let od : szlt d_out = r2 /^ how;
  let r3 : szlt how = r2 %^ how;
  let oh : szlt h_out = r3 /^ w_out;
  let ow : szlt w_out = r3 %^ w_out;

  let kh_kw : sz = kh *^ kw;
  named_mul_value kh kw kh_kw;
  let kd_kh_kw : sz = kd *^ kh_kw;
  named_mul_value kd kh_kw kd_kh_kw;
  flattened_taps_fit cin kd kh kw (SZ.v kh_kw) (SZ.v kd_kh_kw);
  let n_taps : sz = cin *^ kd_kh_kw;
  named_mul_value cin kd_kh_kw n_taps;
  flatten_taps cin kd kh kw (SZ.v kh_kw) (SZ.v kd_kh_kw) (SZ.v n_taps);

  let od_s : sz = od *^ stride;
  let oh_s : sz = oh *^ stride;
  let ow_s : sz = ow *^ stride;

  let mut acc : et = zero;
  let mut k : sz = 0sz;

  while (!k <^ n_taps)
    invariant
      exists* (vk : sz{SZ.v vk <= cin * kd * kh * kw}).
        k |-> vk **
        acc |-> conv3d_partial_at b cin d_in h_in w_in cout kd kh kw
                                   stride (SZ.v pad) d_out h_out w_out
                                   sx sw bi oc od oh ow (SZ.v vk)
    invariant pure (SZ.fits (cin * kd * kh * kw))
    invariant gx |-> Frac (fx /. (b * cout * d_out * h_out * w_out)) sx
    invariant gw |-> Frac (fw /. (b * cout * d_out * h_out * w_out)) sw
    invariant gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias
    invariant gpu
    decreases (cin * kd * kh * kw - SZ.v !k)
  {
    let kk = !k;
    assert pure (SZ.v kk < cin * SZ.v kd_kh_kw);
    div_lt_product (SZ.v kk) cin (SZ.v kd_kh_kw);
    let ic : szlt cin = kk /^ kd_kh_kw;
    let r  : szlt kd_kh_kw = kk %^ kd_kh_kw;
    assert pure (SZ.v r < kd * SZ.v kh_kw);
    div_lt_product (SZ.v r) kd (SZ.v kh_kw);
    let kd_i : szlt kd = r /^ kh_kw;
    let r2  : szlt kh_kw = r %^ kh_kw;
    assert pure (SZ.v r2 < kh * kw);
    div_lt_product (SZ.v r2) kh kw;
    let kh_i : szlt kh = r2 /^ kw;
    let kw_i : szlt kw = r2 %^ kw;

    assert pure (SZ.v kk < cin * kd * kh * kw);
    assert pure (SZ.v ic == SZ.v kk / SZ.v kd_kh_kw);
    assert pure (SZ.v r == SZ.v kk % SZ.v kd_kh_kw);
    assert pure (SZ.v kd_i == SZ.v r / SZ.v kh_kw);
    assert pure (SZ.v r2 == SZ.v r % SZ.v kh_kw);
    assert pure (SZ.v kh_i == SZ.v r2 / kw);
    assert pure (SZ.v kw_i == SZ.v r2 % kw);
    unrank3_from_steps cin kd kh kw (SZ.v kk) (SZ.v kh_kw)
      (SZ.v kd_kh_kw) (SZ.v n_taps) (SZ.v ic) (SZ.v r) (SZ.v kd_i)
      (SZ.v r2) (SZ.v kh_i) (SZ.v kw_i);

    let d_signed = od_s +^ kd_i;
    let h_signed = oh_s +^ kh_i;
    let w_signed = ow_s +^ kw_i;
    let xv = read_x_padded b cin d_in h_in w_in gx bi ic
                           d_signed h_signed w_signed pad;
    let wv = read_w_tap cout cin kd kh kw gw oc ic kd_i kh_i kw_i;
    let prod = mul xv wv;
    let acc0 = !acc;

    (* Establish the step equation: prod equals the lemma's per-tap product. *)
    assert pure (xv ==
      read_padded3 (lseq_to_t5 b cin d_in h_in w_in sx) bi ic
        (SZ.v od * SZ.v stride + SZ.v kd_i * 1 - SZ.v pad)
        (SZ.v oh * SZ.v stride + SZ.v kh_i * 1 - SZ.v pad)
        (SZ.v ow * SZ.v stride + SZ.v kw_i * 1 - SZ.v pad));
    assert pure (wv == t5acc (lseq_to_t5 cout cin kd kh kw sw)
                              oc ic kd_i kh_i kw_i);
    assert pure (SZ.v ic   == unrank3_ic cin kd kh kw (SZ.v kk));
    assert pure (SZ.v kd_i == unrank3_kd cin kd kh kw (SZ.v kk));
    assert pure (SZ.v kh_i == unrank3_kh cin kd kh kw (SZ.v kk));
    assert pure (SZ.v kw_i == unrank3_kw cin kd kh kw (SZ.v kk));
    conv3d_partial_at_step b cin d_in h_in w_in cout kd kh kw
                            stride (SZ.v pad) d_out h_out w_out
                            sx sw bi oc od oh ow (SZ.v kk + 1);

    acc := add acc0 prod;
    assert pure (SZ.v kk < cin * kd * kh * kw);
    decreases_after_increment (cin * kd * kh * kw) (SZ.v kk);
    let knew : sz = !k +^ 1sz;
    assert pure (SZ.v knew == SZ.v kk + 1);
    assert pure (SZ.v knew <= cin * kd * kh * kw);
    k := knew;
  };

  (* Loop exit: !k = vk = n_taps = cin*kd*kh*kw, so acc holds the full
     partial sum [__conv3d_single ... (cin*kd*kh*kw)] (via the [unfold]
     [conv3d_partial_at]).  Adding the bias gives [conv3d_single], which
     equals [conv3d_out_at] applied to [tid] once the kernel-side decode
     of [tid] is shown to match the spec-side decode (modulo
     associativity of [*]). *)
  let bias_v = tensor_read gbias (oc, ());
  let result = add bias_v !acc;

  Math.paren_mul_right h_out w_out 1;
  Math.paren_mul_right d_out h_out w_out;
  Math.paren_mul_right cout d_out (h_out * w_out);
  Math.paren_mul_right (cout * d_out) h_out w_out;
  Math.paren_mul_right cout (d_out * h_out) w_out;
  assert pure (SZ.v how  == h_out * w_out);
  assert pure (SZ.v dhw  == d_out * h_out * w_out);
  assert pure (SZ.v cdhw == cout * d_out * h_out * w_out);
  assert pure (SZ.v bi == SZ.v tid / (cout * d_out * h_out * w_out));
  assert pure (SZ.v r1 == SZ.v tid % (cout * d_out * h_out * w_out));
  assert pure (SZ.v oc == SZ.v r1 / (d_out * h_out * w_out));
  assert pure (SZ.v r2 == SZ.v r1 % (d_out * h_out * w_out));
  assert pure (SZ.v od == SZ.v r2 / (h_out * w_out));
  assert pure (SZ.v r3 == SZ.v r2 % (h_out * w_out));
  assert pure (SZ.v oh == SZ.v r3 / w_out);
  assert pure (SZ.v ow == SZ.v r3 % w_out);

  assert pure (result ==
    conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                  d_out h_out w_out sx sw sbias tid);
  tensor_write_cell gy (tid, ()) result
}

#pop-options

#push-options "--z3rlimit 50"

(* Ghost setup: factor the launcher's full-permission frame into N per-thread
   slices.  Mirrors [Kuiper.Kernel.Conv1D.Naive.conv1d_naive_setup], generalised
   to the three read-only fractional arrays ([gx], [gw], [gbias]) plus a
   full-permission output array ([gy]) exploded into per-cell permissions, with
   the per-thread bound now [b * cout * d_out * h_out * w_out]. *)
ghost
fn conv3d_naive_setup
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (stride : szp)
  (d_out h_out w_out : szp)
  (#lx : layout1 (b * cin * d_in * h_in * w_in))
  (#lw : layout1 (cout * cin * kd * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * d_out * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*cin*d_in*h_in*w_in)))
  (#sw : erased (chest1 et (cout*cin*kd*kh*kw)))
  (#sbias : erased (chest1 et cout))
  (#sy0 : erased (chest1 et (b*cout*d_out*h_out*w_out)))
  (#fx #fw #fb : perm)
  (#_ : squash (conv3d_size_req b cin d_in h_in w_in cout kd kh kw stride
                                d_out h_out w_out))
  ()
  norewrite
  requires
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw) **
    (gbias |-> Frac fb sbias) **
    (gy |-> sy0)
  ensures
    (forall+ (tid : natlt (b *^ cout *^ d_out *^ h_out *^ w_out)).
       kpre #et b cin d_in h_in w_in cout kd kh kw d_out h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw sbias sy0 fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
{
  tensor_pts_to_ref gy;
  tensor_share_n gx (b * cout * d_out * h_out * w_out);
  tensor_share_n gw (b * cout * d_out * h_out * w_out);
  tensor_share_n gbias (b * cout * d_out * h_out * w_out);
  tensor_explode gy;
  forevery_iso (abs_bij #(b * cout * d_out * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * cout * d_out * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc sy0 i);
  forevery_ext
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc sy0 (abs_bij.gg i))
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * cout * d_out * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias)
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * cout * d_out * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * cout * d_out * h_out * w_out)) sw)
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_zip
    (fun (_ : natlt (b * cout * d_out * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * cout * d_out * h_out * w_out)) sx)
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * cout * d_out * h_out * w_out)) sw) **
       (gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (b * cout * d_out * h_out * w_out)
                   (SZ.v (b *^ cout *^ d_out *^ h_out *^ w_out));
  ()
}

(* Ghost teardown: gather N per-thread slices back into the launcher
   postcondition.  Symmetric inverse of [conv3d_naive_setup]. *)
ghost
fn conv3d_naive_teardown
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (stride : szp) (pad : sz)
  (d_out h_out w_out : szp)
  (#lx : layout1 (b * cin * d_in * h_in * w_in))
  (#lw : layout1 (cout * cin * kd * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * d_out * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*cin*d_in*h_in*w_in)))
  (#sw : erased (chest1 et (cout*cin*kd*kh*kw)))
  (#sbias : erased (chest1 et cout))
  (#fx #fw #fb : perm)
  (#_ : squash (conv3d_size_req b cin d_in h_in w_in cout kd kh kw stride
                                d_out h_out w_out))
  ()
  norewrite
  requires
    (forall+ (tid : natlt (b *^ cout *^ d_out *^ h_out *^ w_out)).
       kpost #et b cin d_in h_in w_in cout kd kh kw stride pad
             d_out h_out w_out
             #lx #lw #lbias #ly
             gx gw gbias gy sx sw sbias fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*d_out*h_out*w_out)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*d_out*h_out*w_out}).
               acc1 sy tid ==
               conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                             d_out h_out w_out sx sw sbias tid))
{
  forevery_rw_size (SZ.v (b *^ cout *^ d_out *^ h_out *^ w_out))
                   (b * cout * d_out * h_out * w_out)
    #(kpost #et b cin d_in h_in w_in cout kd kh kw stride pad
            d_out h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw sbias fx fw fb);
  forevery_unzip
    (fun (_ : natlt (b * cout * d_out * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * cout * d_out * h_out * w_out)) sx)
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * cout * d_out * h_out * w_out)) sw) **
       (gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> conv3d_out_at b cin d_in h_in w_in cout kd kh kw
                                    stride pad d_out h_out w_out
                                    sx sw sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * d_out * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * cout * d_out * h_out * w_out)) sw)
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> conv3d_out_at b cin d_in h_in w_in cout kd kh kw
                                    stride pad d_out h_out w_out
                                    sx sw sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * d_out * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * cout * d_out * h_out * w_out)) sbias)
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       Cell gy (idx1 i) |-> conv3d_out_at b cin d_in h_in w_in cout kd kh kw
                                   stride pad d_out h_out w_out
                                   sx sw sbias i);
  tensor_gather_n gx (b * cout * d_out * h_out * w_out);
  tensor_gather_n gw (b * cout * d_out * h_out * w_out);
  tensor_gather_n gbias (b * cout * d_out * h_out * w_out);
  let sy : erased (chest1 et (b * cout * d_out * h_out * w_out)) =
    hide (mk1
            (fun (tid : nat{tid < b * cout * d_out * h_out * w_out}) ->
               conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                             d_out h_out w_out sx sw sbias tid));
  forevery_ext
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       Cell gy (idx1 i) |-> conv3d_out_at b cin d_in h_in w_in cout kd kh kw
                                   stride pad d_out h_out w_out
                                   sx sw sbias i)
    (fun (i : natlt (b * cout * d_out * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc (reveal sy) (abs_bij.gg i));
  forevery_iso_back (abs_bij #(b * cout * d_out * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * cout * d_out * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc (reveal sy) i);
  tensor_implode gy;
  ()
}

#pop-options

#push-options "--z3rlimit 50 --fuel 2 --ifuel 2"

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (stride : szp) (pad : sz)
  (d_out h_out w_out : szp)
  (#lx : layout1 (b * cin * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*cin*d_in*h_in*w_in)))
  (#sw : erased (chest1 et (cout*cin*kd*kh*kw)))
  (#sbias : erased (chest1 et cout))
  (#sy0 : erased (chest1 et (b*cout*d_out*h_out*w_out)))
  (#fx #fw #fb : perm)
  (#_ : squash (is_global gx /\ is_global gw /\
                is_global gbias /\ is_global gy /\
                conv3d_size_req b cin d_in h_in w_in cout kd kh kw stride
                                d_out h_out w_out))
  : kernel_desc
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 et (b*cout*d_out*h_out*w_out)).
         (gy |-> sy) **
         pure (forall (tid : nat{tid < b*cout*d_out*h_out*w_out}).
                 acc1 sy tid ==
                 conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                               d_out h_out w_out sx sw sbias tid)))
=
{
  nthr = b *^ cout *^ d_out *^ h_out *^ w_out;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = conv3d_naive_setup b cin d_in h_in w_in cout kd kh kw stride
                                d_out h_out w_out gx gw gbias gy;
  teardown = conv3d_naive_teardown b cin d_in h_in w_in cout kd kh kw stride pad
                                   d_out h_out w_out gx gw gbias gy;
  kpre  = kpre #et b cin d_in h_in w_in cout kd kh kw d_out h_out w_out
               #lx #lw #lbias #ly gx gw gbias gy sx sw sbias sy0 fx fw fb;
  kpost = kpost #et b cin d_in h_in w_in cout kd kh kw stride pad
                d_out h_out w_out
                #lx #lw #lbias #ly gx gw gbias gy sx sw sbias fx fw fb;
  f = kf b cin d_in h_in w_in cout kd kh kw stride pad d_out h_out w_out
        gx gw gbias gy;
  (* Inherited tree-wide debt (sendability of compound per-thread slprops). *)
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn conv3d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (stride : szp) (pad : sz)
  (d_out h_out w_out : szp)
  (#lx : layout1 (b * cin * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : erased (chest1 et (b*cin*d_in*h_in*w_in)))
  (#sw : erased (chest1 et (cout*cin*kd*kh*kw)))
  (#sbias : erased (chest1 et cout))
  (#sy0 : erased (chest1 et (b*cout*d_out*h_out*w_out)))
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
{
  launch_sync (kdesc b cin d_in h_in w_in cout kd kh kw stride pad
                     d_out h_out w_out gx gw gbias gy)
}
