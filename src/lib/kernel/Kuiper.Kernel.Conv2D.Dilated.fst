module Kuiper.Kernel.Conv2D.Dilated

(* Implementation of [Kuiper.Kernel.Conv2D.Dilated].  See the [.fsti]
   for the contract.  This file is a structural sibling of
   [Kuiper.Kernel.Conv2D.Naive]: per-axis (sh, sw, ph, pw, dh, dw)
   instead of (stride, pad, dilation=1).

   Setup, teardown, and kpre/kpost sendability are all discharged at
   the [kdesc] level.
   The per-thread loop's spec connection to
   [Kuiper.Spec.Conv2DDilated.__conv2dd_single] is discharged via the
   [conv2dd_partial_at] / [conv2dd_partial_at_step] pattern (mirrors
   [Kuiper.Kernel.Conv1D.Naive]). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv2D
open Kuiper.Spec.Conv2DDilated
open Kuiper.Kernel.Conv2D.Naive { lseq_to_t4 }
open Kuiper.Bijection { ( =~ ) }
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

let flatten4_index_bound
  (b cin h w : pos)
  (bi ic hi wi : nat)
  : Lemma
      (requires bi < b /\ ic < cin /\ hi < h /\ wi < w)
      (ensures (((bi * cin + ic) * h + hi) * w + wi) < b * cin * h * w)
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
  (cin kh kw kh_kw n_taps : nat)
  : Lemma
      (requires kh_kw == kh * kw /\ n_taps == cin * kh_kw)
      (ensures n_taps == cin * kh * kw)
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


(* Per-thread pre/post predicates. *)

unfold
let kpre
  (#et:Type) {| scalar et |}
  (b cin h_in w_in cout : pos) (kh kw : pos)
  (h_out w_out : pos)
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cout * cin * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_ : chest1 et (cout*cin*kh*kw))
  (sbias : chest1 et cout)
  (sy0 : chest1 et (b*cout*h_out*w_out))
  (fx fw fb : perm)
  (tid : natlt (b * cout * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_ **
    gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> acc1 sy0 tid

unfold
let kpost
  (#et:Type) {| scalar et |}
  (b cin h_in w_in cout : pos) (kh kw : pos)
  (sh sw : pos) (ph pw : nat) (dh dw : pos)
  (h_out w_out : pos)
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cout * cin * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_ : chest1 et (cout*cin*kh*kw))
  (sbias : chest1 et cout)
  (fx fw fb : perm)
  (tid : natlt (b * cout * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_ **
    gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                   h_out w_out sx sw_ sbias tid

#push-options "--z3rlimit 120"

(* Inner-loop helper: read a tap from x at (bi, ic, h_int, w_int)
   with zero-padded out-of-range guards, where
     h_int = oh*sh + kh_i*dh - ph,  w_int = ow*sw + kw_i*dw - pw.
   We pass already-summed offsets (oh*sh + kh_i*dh) and (ow*sw + kw_i*dw)
   as [h_signed], [w_signed] in u32; the per-axis pad subtraction is
   done locally with branch guards. *)
inline_for_extraction noextract
fn read_x_padded_axis
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in : szp)
  (#lx : layout1 (b * cin * h_in * w_in)) {| ctlayout lx |}
  (gx : array1 et lx)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#fx : perm)
  (bi : szlt b)
  (ic : szlt cin)
  (h_signed : sz) (* oh*sh + kh_i*dh, in u32 *)
  (w_signed : sz) (* ow*sw + kw_i*dw, in u32 *)
  (ph : sz) (pw : sz)
  (#_ : squash (SZ.fits (b * cin * h_in * w_in)))
  preserves
    gpu **
    gx |-> Frac fx sx
  returns
    v : et
  ensures
    pure (
      let h_int : int = SZ.v h_signed - SZ.v ph in
      let w_int : int = SZ.v w_signed - SZ.v pw in
      v == (if 0 <= h_int && h_int < h_in &&
               0 <= w_int && w_int < w_in
            then acc1 sx (((bi * cin + ic) * h_in + h_int) * w_in + w_int)
            else zero))
{
  if (ph <=^ h_signed && pw <=^ w_signed) {
    let hi = h_signed -^ ph;
    let wi = w_signed -^ pw;
    if (hi <^ h_in && wi <^ w_in) {
      flatten4_index_bound b cin h_in w_in (SZ.v bi) (SZ.v ic)
        (SZ.v hi) (SZ.v wi);
      let bcin = bi *^ cin;
      let bcin_ic = bcin +^ ic;
      let bcin_ic_h = bcin_ic *^ h_in;
      let bcin_ic_h_hi = bcin_ic_h +^ hi;
      let bcin_ic_h_hi_w = bcin_ic_h_hi *^ w_in;
      let flat : szlt (b * cin * h_in * w_in) = bcin_ic_h_hi_w +^ wi;
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

(* Read a weight tap [(oc, ic, kh_i, kw_i)] from the flat weight array. *)
inline_for_extraction noextract
fn read_w_tap
  (#et : Type0) {| scalar et |}
  (cout cin kh kw : szp)
  (#lw : layout1 (cout * cin * kh * kw)) {| ctlayout lw |}
  (gw : array1 et lw)
  (#sw_ : chest1 et (cout*cin*kh*kw))
  (#fw : perm)
  (oc : szlt cout) (ic : szlt cin)
  (kh_i : szlt kh) (kw_i : szlt kw)
  (#_ : squash (SZ.fits (cout * cin * kh * kw)))
  preserves
    gpu **
    gw |-> Frac fw sw_
  returns
    v : et
  ensures
    pure (v == acc1 sw_ (((oc * cin + ic) * kh + kh_i) * kw + kw_i))
{
  Math.lemma_mult_lt_right cin oc cout;
  Math.lemma_mult_le_right kh (oc * cin + ic + 1) (cout * cin);
  Math.lemma_mult_le_right kw ((oc * cin + ic) * kh + kh_i + 1) (cout * cin * kh);
  let p1 = oc *^ cin +^ ic;
  let p2 = p1 *^ kh +^ kh_i;
  let flat : szlt (cout * cin * kh * kw) = p2 *^ kw +^ kw_i;
  tensor_read gw (flat, ())
}

#pop-options

#push-options "--z3rlimit 200 --fuel 2 --ifuel 1"

(* Local helper: partial conv2dd sum over the linearised (ic, kh_i, kw_i)
   index up to [to], with all parameters explicit.  Used as the loop-
   invariant predicate for [kf]'s accumulator.  Same pattern as
   [Kuiper.Kernel.Conv1D.Naive.conv1d_partial_at]: this wrapper avoids
   unification difficulties from passing the [val]-only
   [__conv2dd_single] directly inside a Pulse [exists*] slprop. *)
unfold
let conv2dd_partial_at
  (#et : Type) {| scalar et |}
  (b cin h_in w_in cout : pos)
  (kh kw : pos) (sh sw : pos) (ph pw : nat) (dh dw : pos)
  (h_out w_out : nat)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_ : chest1 et (cout*cin*kh*kw))
  (bi : natlt b) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= cin * kh * kw})
  : GTot et
  = __conv2dd_single kh kw sh sw ph pw dh dw
      (lseq_to_t4 b cin h_in w_in sx)
      (lseq_to_t4 cout cin kh kw sw_)
      bi oc oh ow to

(* Step lemma for [conv2dd_partial_at]: extends the partial sum by one tap. *)
let conv2dd_partial_at_step
  (#et : Type) {| scalar et |}
  (b cin h_in w_in cout : pos)
  (kh kw : pos) (sh sw : pos) (ph pw : nat) (dh dw : pos)
  (h_out w_out : nat)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_ : chest1 et (cout*cin*kh*kw))
  (bi : natlt b) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank_ic cin kh kw i in
      let kh_i = unrank_kh cin kh kw i in
      let kw_i = unrank_kw cin kh kw i in
      let h_idx : int = oh * sh + kh_i * dh - ph in
      let w_idx : int = ow * sw + kw_i * dw - pw in
      conv2dd_partial_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                         h_out w_out sx sw_ bi oc oh ow to ==
      add (conv2dd_partial_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                              h_out w_out sx sw_ bi oc oh ow (to - 1))
          (mul (read_padded (lseq_to_t4 b cin h_in w_in sx) bi ic h_idx w_idx)
               (tacc (lseq_to_t4 cout cin kh kw sw_) oc ic kh_i kw_i))))
  = __conv2dd_single_lemma cin kh kw sh sw ph pw dh dw
      (lseq_to_t4 b cin h_in w_in sx)
      (lseq_to_t4 cout cin kh kw sw_)
      bi oc oh ow to

(* Per-thread conv body: decode tid, run the inner accumulator loop,
   add bias, write to output cell.  Spec correctness against
   [Kuiper.Spec.Conv2DDilated.conv2dd_single] is discharged via the
   [conv2dd_partial_at] / [conv2dd_partial_at_step] pattern (mirrors
   [Kuiper.Kernel.Conv2D.Naive.kf]); no [assume pure] is used in the
   per-thread body. *)
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (sh sw : szp) (ph pw : sz)
  (dh dw : szp)
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
  (#sw_ : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (b * cout * h_out * w_out > 0))
  (#_ : squash (SZ.fits (cin * kh * kw) /\
                SZ.fits (h_out * w_out) /\
                SZ.fits (cout * h_out * w_out) /\
                SZ.fits (b * cin * h_in * w_in) /\
                SZ.fits (cout * cin * kh * kw) /\
                SZ.fits (h_out * sh + kh * dh) /\
                SZ.fits (w_out * sw + kw * dw)))
  (tid : szlt (b * cout * h_out * w_out))
  ()
  norewrite
  requires
    gpu **
    kpre #et b cin h_in w_in cout kh kw h_out w_out #lx #lw #lbias #ly
         gx gw gbias gy sx sw_ sbias sy0 fx fw fb tid
  ensures
    gpu **
    kpost #et b cin h_in w_in cout kh kw sh sw ph pw dh dw h_out w_out
          #lx #lw #lbias #ly
          gx gw gbias gy sx sw_ sbias fx fw fb tid
{
  (* The per-thread body proves [result == conv2dd_out_at ...] via a loop
     invariant tracking [acc == conv2dd_partial_at ... (SZ.v k)] and the
     step lemma [conv2dd_partial_at_step] (which wraps the spec-level
     [__conv2dd_single_lemma]).  Setup, teardown, and sendability are
     discharged at the [kdesc] level (see below). *)
  let how : sz = h_out *^ w_out;
  let chow : sz = cout *^ how;
  let bi : szlt b = tid /^ chow;
  let r1 : szlt chow = tid %^ chow;
  let oc : szlt cout = r1 /^ how;
  let r2 : szlt how = r1 %^ how;
  let oh : szlt h_out = r2 /^ w_out;
  let ow : szlt w_out = r2 %^ w_out;

  let kh_kw : sz = kh *^ kw;
  let n_taps : sz = cin *^ kh_kw;
  named_mul_value kh kw kh_kw;
  named_mul_value cin kh_kw n_taps;
  flatten_taps cin kh kw (SZ.v kh_kw) (SZ.v n_taps);

  let oh_s : sz = oh *^ sh;
  let ow_s : sz = ow *^ sw;

  let mut acc : et = zero;
  let mut k : sz = 0sz;

  while (!k <^ n_taps)
    invariant
      exists* (vk : sz{SZ.v vk <= cin * kh * kw}).
        k |-> vk **
        acc |-> conv2dd_partial_at b cin h_in w_in cout kh kw sh sw ph pw
                  dh dw h_out w_out sx sw_ bi oc oh ow (SZ.v vk)
    invariant gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx
    invariant gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_
    invariant gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias
    invariant gpu
    decreases (cin * kh * kw - SZ.v !k)
  {
    let kk_v = !k;
    assert pure (SZ.v kk_v < cin * kh * kw);
    let ic : szlt cin = kk_v /^ kh_kw;
    let r : szlt kh_kw = kk_v %^ kh_kw;
    let kh_i : szlt kh = r /^ kw;
    let kw_i : szlt kw = r %^ kw;

    let kh_dh : sz = kh_i *^ dh;
    let kw_dw : sz = kw_i *^ dw;
    let h_signed = oh_s +^ kh_dh;
    let w_signed = ow_s +^ kw_dw;
    let xv = read_x_padded_axis b cin h_in w_in gx bi ic h_signed w_signed ph pw;
    let wv = read_w_tap cout cin kh kw gw oc ic kh_i kw_i;
    let prod = mul xv wv;
    let acc0 = !acc;
    (* Establish the step equation: prod equals the lemma's per-tap product. *)
    assert pure (xv ==
      read_padded (lseq_to_t4 b cin h_in w_in sx) bi ic
        (oh * sh + kh_i * dh - ph) (ow * sw + kw_i * dw - pw));
    assert pure (wv == tacc (lseq_to_t4 cout cin kh kw sw_) oc ic kh_i kw_i);
    assert pure (SZ.v ic == unrank_ic cin kh kw (SZ.v kk_v));
    assert pure (SZ.v kh_i == unrank_kh cin kh kw (SZ.v kk_v));
    assert pure (SZ.v kw_i == unrank_kw cin kh kw (SZ.v kk_v));
    conv2dd_partial_at_step b cin h_in w_in cout kh kw sh sw ph pw dh dw
      h_out w_out sx sw_ bi oc oh ow (SZ.v kk_v + 1);
    acc := add acc0 prod;
    decreases_after_increment (cin * kh * kw) (SZ.v kk_v);
    let next_k = !k +^ 1sz;
    assert pure (SZ.v next_k == SZ.v kk_v + 1);
    k := next_k;
  };

  let bias_v = tensor_read gbias (oc, ());
  let result = add bias_v !acc;
  tensor_write_cell gy (tid, ()) result
}

#pop-options

#push-options "--z3rlimit 60"

(* Ghost setup: factor the launcher's full-permission frame into N per-thread
   slices.  Mirrors [Kuiper.Kernel.Conv2D.Naive.conv2d_naive_setup], adapted
   for per-axis stride/pad/dilation. *)
ghost
fn conv2d_dilated_setup
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (sh sw : szp) (ph pw : sz)
  (dh dw : szp)
  (h_out w_out : szp)
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cout * cin * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw_ : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (conv2dd_size_req b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                 h_out w_out))
  ()
  norewrite
  requires
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw_) **
    (gbias |-> Frac fb sbias) **
    (gy |-> sy0)
  ensures
    (forall+ (tid : natlt (b *^ cout *^ h_out *^ w_out)).
       kpre #et b cin h_in w_in cout kh kw h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw_ sbias sy0 fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
{
  tensor_pts_to_ref gy;
  tensor_share_n gx (b * cout * h_out * w_out);
  tensor_share_n gw (b * cout * h_out * w_out);
  tensor_share_n gbias (b * cout * h_out * w_out);
  tensor_explode gy;
  forevery_iso (abs_bij #(b * cout * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * cout * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc sy0 i);
  forevery_ext
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc sy0 (abs_bij.gg i))
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_zip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_) **
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (b * cout * h_out * w_out)
                   (SZ.v (b *^ cout *^ h_out *^ w_out));
  ()
}

(* Ghost teardown: gather N per-thread slices back into the launcher
   postcondition.  Symmetric inverse of [conv2d_dilated_setup]. *)
ghost
fn conv2d_dilated_teardown
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (sh sw : szp) (ph pw : sz)
  (dh dw : szp)
  (h_out w_out : szp)
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cout * cin * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw_ : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#fx #fw #fb : perm)
  (#_ : squash (conv2dd_size_req b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                 h_out w_out))
  ()
  norewrite
  requires
    (forall+ (tid : natlt (b *^ cout *^ h_out *^ w_out)).
       kpost #et b cin h_in w_in cout kh kw sh sw ph pw dh dw h_out w_out
             #lx #lw #lbias #ly
             gx gw gbias gy sx sw_ sbias fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw_) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*h_out*w_out)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
               acc1 sy tid ==
               conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                              h_out w_out sx sw_ sbias tid))
{
  forevery_rw_size (SZ.v (b *^ cout *^ h_out *^ w_out))
                   (b * cout * h_out * w_out)
    #(kpost #et b cin h_in w_in cout kh kw sh sw ph pw dh dw h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw_ sbias fx fw fb);
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_) **
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                     h_out w_out sx sw_ sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                     h_out w_out sx sw_ sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (idx1 i) |-> conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                    h_out w_out sx sw_ sbias i);
  tensor_gather_n gx (b * cout * h_out * w_out);
  tensor_gather_n gw (b * cout * h_out * w_out);
  tensor_gather_n gbias (b * cout * h_out * w_out);
  let sy : chest1 et (b * cout * h_out * w_out) =
    hide (mk1
            (fun (tid : nat{tid < b * cout * h_out * w_out}) ->
               conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                              h_out w_out sx sw_ sbias tid));
  forevery_ext
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (idx1 i) |-> conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                    h_out w_out sx sw_ sbias i)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc (reveal sy) (abs_bij.gg i));
  forevery_iso_back (abs_bij #(b * cout * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * cout * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc (reveal sy) i);
  tensor_implode gy;
  ()
}

#pop-options

#push-options "--z3rlimit 200 --fuel 2 --ifuel 2"

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (sh sw : szp) (ph pw : sz)
  (dh dw : szp)
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
  (#sw_ : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (is_global gx /\ is_global gw /\
                is_global gbias /\ is_global gy /\
                conv2dd_size_req b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                 h_out w_out))
  : kernel_desc
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw_) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw_) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 et (b*cout*h_out*w_out)).
         (gy |-> sy) **
         pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
                 acc1 sy tid ==
                 conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                h_out w_out sx sw_ sbias tid)))
=
{
  nthr = b *^ cout *^ h_out *^ w_out;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = conv2d_dilated_setup b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                  h_out w_out gx gw gbias gy;
  teardown = conv2d_dilated_teardown b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                     h_out w_out gx gw gbias gy;
  kpre  = kpre #et b cin h_in w_in cout kh kw h_out w_out #lx #lw #lbias #ly gx gw gbias gy sx sw_ sbias sy0 fx fw fb;
  kpost = kpost #et b cin h_in w_in cout kh kw sh sw ph pw dh dw h_out w_out #lx #lw #lbias #ly gx gw gbias gy sx sw_ sbias fx fw fb;
  f = kf b cin h_in w_in cout kh kw sh sw ph pw dh dw h_out w_out gx gw gbias gy;
  (* Inherited tree-wide debt (sendability of compound per-thread slprops). *)
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn conv2d_dilated_gpu
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (sh sw : szp) (ph pw : sz)
  (dh dw : szp)
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
  (#sw_ : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  norewrite
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          conv2dd_size_req b cin h_in w_in cout kh kw sh sw ph pw dh dw
                           h_out w_out)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*h_out*w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
               acc1 sy tid ==
               conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                              h_out w_out sx sw_ sbias tid))
{
  launch_sync (kdesc b cin h_in w_in cout kh kw sh sw ph pw dh dw h_out w_out
                     gx gw gbias gy)
}
