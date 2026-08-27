module Kuiper.Kernel.ConvT2D.Naive

(* Implementation of [Kuiper.Kernel.ConvT2D.Naive].  See [.fsti] for
   the contract.  The kernel computes one output pixel per thread; the
   inner accumulation over the [(ic, kh_i, kw_i)] taps is a single
   while-loop matched up to [Kuiper.Spec.ConvTranspose2D.__convT2d_single]
   via the [conv1d_partial_at] proof pattern (loop invariant tracks
   [acc == convT2d_partial_at ... (SZ.v k)]; step lemma extends by one
   tap).  Setup, teardown, and kpre/kpost sendability are all
   discharged at the [kdesc] level. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv2D
open Kuiper.Spec.ConvTranspose2D
open FStar.FunctionalExtensionality { (^->>) }
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


let lseq_to_t4
  (#et:Type) (d0 d1 d2 d3 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3))
  : etensor4 et d0 d1 d2 d3
  = mkT4 (fun i j k l -> acc1 s (((i * d1 + j) * d2 + k) * d3 + l))

let lseq_to_t4_index
  (#et:Type) (d0 d1 d2 d3 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3))
  (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3)
  : Lemma (tacc (lseq_to_t4 d0 d1 d2 d3 s) i j k l ==
           acc1 s (((i * d1 + j) * d2 + k) * d3 + l))
          [SMTPat (tacc (lseq_to_t4 d0 d1 d2 d3 s) i j k l)]
  = ()

(* Per-thread pre/post predicates. *)

unfold
let kpre
  (#et:Type) {| scalar et |}
  (b cin h_in w_in cout : pos) (kh kw : pos)
  (h_out w_out : pos)
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cin * cout * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_l : chest1 et (cin*cout*kh*kw))
  (sbias : chest1 et cout)
  (sy0 : chest1 et (b*cout*h_out*w_out))
  (fx fw fb : perm)
  (tid : natlt (b * cout * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_l **
    gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> acc1 sy0 tid

unfold
let kpost
  (#et:Type) {| scalar et |}
  (b cin h_in w_in cout : pos) (kh kw : pos)
  (sh sw : pos) (ph pw : nat) (dh dw : pos)
  (h_out w_out : pos)
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cin * cout * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_l : chest1 et (cin*cout*kh*kw))
  (sbias : chest1 et cout)
  (fx fw fb : perm)
  (tid : natlt (b * cout * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_l **
    gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> convT2d_out_at b cin h_in w_in cout kh kw
                                   sh sw ph pw dh dw
                                   h_out w_out sx sw_l sbias tid

#push-options "--z3rlimit 80"

(* Inner-loop helper: read tap from x with strided + zero-padded
   ConvTranspose semantics.  Reads x[bi, ic, num_h/sh, num_w/sw] iff
   num_h, num_w >= 0, divisible by sh/sw, and within range. *)
inline_for_extraction noextract
fn read_x_strided_pad
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in : szp)
  (#lx : layout1 (b * cin * h_in * w_in)) {| ctlayout lx |}
  (gx : array1 et lx)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#fx : perm)
  (bi : szlt b)
  (ic : szlt cin)
  (oh_ph : sz)   (* oh + ph *)
  (ow_pw : sz)   (* ow + pw *)
  (kh_dh : sz)   (* kh_i * dh *)
  (kw_dw : sz)   (* kw_i * dw *)
  (sh sw : szp)
  (#_ : squash (SZ.fits (b * cin * h_in * w_in)))
  preserves
    gpu **
    gx |-> Frac fx sx
  returns
    v : et
  ensures
    pure (
      let h_num : int = SZ.v oh_ph - SZ.v kh_dh in
      let w_num : int = SZ.v ow_pw - SZ.v kw_dw in
      v == (if h_num >= 0 && w_num >= 0
              && h_num % SZ.v sh = 0 && w_num % SZ.v sw = 0
              && h_num / SZ.v sh < h_in && w_num / SZ.v sw < w_in
            then acc1 sx
                 (((bi * cin + ic) * h_in + h_num / SZ.v sh) * w_in
                    + w_num / SZ.v sw)
            else zero))
{
  if (oh_ph >=^ kh_dh && ow_pw >=^ kw_dw) {
    let h_num = oh_ph -^ kh_dh;
    let w_num = ow_pw -^ kw_dw;
    let h_rem = h_num %^ sh;
    let w_rem = w_num %^ sw;
    if (h_rem = 0sz && w_rem = 0sz) {
      let hi = h_num /^ sh;
      let wi = w_num /^ sw;
      if (hi <^ h_in && wi <^ w_in) {
        let p1 = bi *^ cin +^ ic;
        let p2 = p1 *^ h_in +^ hi;
        let flat : szlt (b * cin * h_in * w_in) = p2 *^ w_in +^ wi;
        let v = tensor_read gx (flat, ());
        v
      } else {
        zero
      }
    } else {
      zero
    }
  } else {
    zero
  }
}

#pop-options

#push-options "--z3rlimit 80"

(* Read a weight tap [(ic, oc, kh_i, kw_i)] from the flat weight array.
   Note: ConvT layout is (cin, cout, kh, kw) — different from forward conv. *)
inline_for_extraction noextract
fn read_w_tap_t
  (#et : Type0) {| scalar et |}
  (cin cout kh kw : szp)
  (#lw : layout1 (cin * cout * kh * kw)) {| ctlayout lw |}
  (gw : array1 et lw)
  (#sw_l : chest1 et (cin*cout*kh*kw))
  (#fw : perm)
  (ic : szlt cin) (oc : szlt cout)
  (kh_i : szlt kh) (kw_i : szlt kw)
  (#_ : squash (SZ.fits (cin * cout * kh * kw)))
  preserves
    gpu **
    gw |-> Frac fw sw_l
  returns
    v : et
  ensures
    pure (v == acc1 sw_l (((ic * cout + oc) * kh + kh_i) * kw + kw_i))
{
  Math.lemma_mult_lt_right cout ic cin;
  Math.lemma_mult_le_right kh (ic * cout + oc + 1) (cin * cout);
  Math.lemma_mult_le_right kw ((ic * cout + oc) * kh + kh_i + 1) (cin * cout * kh);
  let p1 = ic *^ cout +^ oc;
  let p2 = p1 *^ kh +^ kh_i;
  let flat : szlt (cin * cout * kh * kw) = p2 *^ kw +^ kw_i;
  tensor_read gw (flat, ())
}

#pop-options

#push-options "--z3rlimit 400 --fuel 2 --ifuel 1"

(* Local helper: partial convT2d sum over the linearised (ic, kh, kw)
   index up to [to], with all parameters explicit.  Used as the loop-
   invariant predicate for [kf]'s accumulator.  Wrapping the val-only
   [__convT2d_single] avoids unification difficulties when the spec
   appears inside a Pulse [exists*] slprop. *)
unfold
let convT2d_partial_at
  (#et : Type) {| scalar et |}
  (b cin h_in w_in cout : pos)
  (kh kw : pos)
  (sh sw : pos) (ph pw : nat) (dh dw : pos)
  (h_out w_out : nat)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_l : chest1 et (cin*cout*kh*kw))
  (sbias : chest1 et cout)
  (bi : natlt b) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= cin * kh * kw})
  : GTot et
  = __convT2d_single kh kw sh sw ph pw dh dw
      (lseq_to_t4 b cin h_in w_in sx)
      (lseq_to_t4 cin cout kh kw sw_l)
      bi oc oh ow to

(* Step lemma for [convT2d_partial_at]: extends the partial sum by one tap. *)
let convT2d_partial_at_step
  (#et : Type) {| scalar et |}
  (b cin h_in w_in cout : pos)
  (kh kw : pos)
  (sh sw : pos) (ph pw : nat) (dh dw : pos)
  (h_out w_out : nat)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw_l : chest1 et (cin*cout*kh*kw))
  (sbias : chest1 et cout)
  (bi : natlt b) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank_ic cin kh kw i in
      let kh_i = unrank_kh cin kh kw i in
      let kw_i = unrank_kw cin kh kw i in
      let h_num : int = oh + ph - kh_i * dh in
      let w_num : int = ow + pw - kw_i * dw in
      convT2d_partial_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                         h_out w_out sx sw_l sbias bi oc oh ow to ==
      add (convT2d_partial_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias bi oc oh ow (to - 1))
          (mul (read_strided_padded_2d
                  (lseq_to_t4 b cin h_in w_in sx) bi ic sh sw h_num w_num)
               (tacc (lseq_to_t4 cin cout kh kw sw_l) ic oc kh_i kw_i))))
  = __convT2d_single_lemma cin kh kw sh sw ph pw dh dw
      (lseq_to_t4 b cin h_in w_in sx)
      (lseq_to_t4 cin cout kh kw sw_l)
      bi oc oh ow to

(* Per-thread ConvT body: decode tid, run inner accumulator loop,
   add bias, write output cell.  Spec-connection is now discharged via
   the [convT2d_partial_at] loop invariant; setup, teardown, and
   sendability are discharged at the [kdesc] level (see below). *)
inline_for_extraction noextract
fn kf
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
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw_l : chest1 et (cin*cout*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (b * cout * h_out * w_out > 0))
  (#_ : squash (SZ.fits (cin * kh * kw) /\
                SZ.fits (kh * kw) /\
                SZ.fits (h_out * w_out) /\
                SZ.fits (cout * h_out * w_out) /\
                SZ.fits (b * cin * h_in * w_in) /\
                SZ.fits (cin * cout * kh * kw) /\
                SZ.fits (h_out + ph) /\
                SZ.fits (w_out + pw) /\
                SZ.fits (kh * dh) /\
                SZ.fits (kw * dw)))
  (tid : szlt (b * cout * h_out * w_out))
  ()
  norewrite
  requires
    gpu **
    kpre #et b cin h_in w_in cout kh kw h_out w_out
         #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias sy0 fx fw fb tid
  ensures
    gpu **
    kpost #et b cin h_in w_in cout kh kw sh sw ph pw dh dw
          h_out w_out
          #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias fx fw fb tid
{
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

  let oh_ph : sz = oh +^ ph;
  let ow_pw : sz = ow +^ pw;

  let mut acc : et = zero;
  let mut k : sz = 0sz;

  while (!k <^ n_taps)
    invariant
      exists* (vk : sz{SZ.v vk <= cin * kh * kw}).
        k |-> vk **
        acc |-> convT2d_partial_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                  h_out w_out sx sw_l sbias bi oc oh ow (SZ.v vk)
    invariant pure (SZ.fits (cin * kh * kw))
    invariant gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx
    invariant gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_l
    invariant gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias
    invariant gpu
    decreases (cin * kh * kw - SZ.v !k)
  {
    let kk = !k;
    let ic : szlt cin = kk /^ kh_kw;
    let r : szlt kh_kw = kk %^ kh_kw;
    let kh_i : szlt kh = r /^ kw;
    let kw_i : szlt kw = r %^ kw;

    let kh_dh : sz = kh_i *^ dh;
    let kw_dw : sz = kw_i *^ dw;
    let xv = read_x_strided_pad b cin h_in w_in gx bi ic
                                 oh_ph ow_pw kh_dh kw_dw sh sw;
    let wv = read_w_tap_t cin cout kh kw gw ic oc kh_i kw_i;
    let prod = mul xv wv;
    let acc0 = !acc;
    (* Establish the step equation: prod equals the lemma's per-tap product. *)
    Math.paren_mul_right cin kh kw;
    assert pure (SZ.v n_taps == cin * kh * kw);
    assert pure (SZ.v ic == unrank_ic cin kh kw (SZ.v kk));
    assert pure (SZ.v kh_i == unrank_kh cin kh kw (SZ.v kk));
    assert pure (SZ.v kw_i == unrank_kw cin kh kw (SZ.v kk));
    assert pure (xv == read_strided_padded_2d
                         (lseq_to_t4 b cin h_in w_in sx) bi ic sh sw
                         (oh + ph - SZ.v kh_i * dh)
                         (ow + pw - SZ.v kw_i * dw));
    assert pure (wv == tacc (lseq_to_t4 cin cout kh kw sw_l)
                            ic oc kh_i kw_i);
    convT2d_partial_at_step b cin h_in w_in cout kh kw sh sw ph pw dh dw
      h_out w_out sx sw_l sbias bi oc oh ow (SZ.v kk + 1);
    acc := add acc0 prod;
    let knew : sz = !k +^ 1sz;
    assert pure (SZ.v knew <= cin * kh * kw);
    k := knew;
  };

  let bias_v = tensor_read gbias (oc, ());
  let result = add bias_v !acc;
  tensor_write_cell gy (tid, ()) result
}

#pop-options

#push-options "--z3rlimit 80"

(* Ghost setup: factor the launcher's full-permission frame into N per-thread
   slices.  Mirrors [Kuiper.Kernel.Conv1D.Naive.conv1d_naive_setup]. *)
ghost
fn convt2d_naive_setup
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp) (kh kw : szp)
  (sh sw : szp) (ph pw : sz) (dh dw : szp)
  (h_out w_out : szp)
  (nthr : szp { SZ.v nthr == b * cout * h_out * w_out })
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cin * cout * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw_l : chest1 et (cin*cout*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (convT2d_size_req b cin h_in w_in cout kh kw
                                 sh sw ph pw dh dw h_out w_out))
  ()
  norewrite
  requires
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw_l) **
    (gbias |-> Frac fb sbias) **
    (gy |-> sy0)
  ensures
    (forall+ (tid : natlt nthr).
       kpre #et b cin h_in w_in cout kh kw h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw_l sbias sy0 fx fw fb tid) **
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
       gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_l)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_zip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_l) **
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (b * cout * h_out * w_out) nthr;
  ()
}

(* Ghost teardown: gather N per-thread slices back into the launcher
   postcondition.  Symmetric inverse of [convt2d_naive_setup]. *)
ghost
fn convt2d_naive_teardown
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp) (kh kw : szp)
  (sh sw : szp) (ph pw : sz) (dh dw : szp)
  (h_out w_out : szp)
  (nthr : szp { SZ.v nthr == b * cout * h_out * w_out })
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cin * cout * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw_l : chest1 et (cin*cout*kh*kw))
  (#sbias : chest1 et cout)
  (#fx #fw #fb : perm)
  (#_ : squash (convT2d_size_req b cin h_in w_in cout kh kw
                                 sh sw ph pw dh dw h_out w_out))
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpost #et b cin h_in w_in cout kh kw sh sw ph pw dh dw
             h_out w_out
             #lx #lw #lbias #ly
             gx gw gbias gy sx sw_l sbias fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw_l) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*h_out*w_out)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
               acc1 sy tid ==
               convT2d_out_at b cin h_in w_in cout kh kw
                              sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias tid))
{
  forevery_rw_size nthr (b * cout * h_out * w_out)
    #(kpost #et b cin h_in w_in cout kh kw sh sw ph pw dh dw
            h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw_l sbias fx fw fb);
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_l) **
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> convT2d_out_at b cin h_in w_in cout kh kw
                                     sh sw ph pw dh dw
                                     h_out w_out sx sw_l sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw_l)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> convT2d_out_at b cin h_in w_in cout kh kw
                                     sh sw ph pw dh dw
                                     h_out w_out sx sw_l sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (idx1 i) |-> convT2d_out_at b cin h_in w_in cout kh kw
                                    sh sw ph pw dh dw
                                    h_out w_out sx sw_l sbias i);
  tensor_gather_n gx (b * cout * h_out * w_out);
  tensor_gather_n gw (b * cout * h_out * w_out);
  tensor_gather_n gbias (b * cout * h_out * w_out);
  let sy : chest1 et (b * cout * h_out * w_out) =
    hide (mk1
            (fun (tid : nat{tid < b * cout * h_out * w_out}) ->
               convT2d_out_at b cin h_in w_in cout kh kw
                              sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias tid));
  forevery_ext
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (idx1 i) |-> convT2d_out_at b cin h_in w_in cout kh kw
                                    sh sw ph pw dh dw
                                    h_out w_out sx sw_l sbias i)
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
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw_l : chest1 et (cin*cout*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (is_global gx /\ is_global gw /\
                is_global gbias /\ is_global gy /\
                convT2d_size_req b cin h_in w_in cout kh kw
                                 sh sw ph pw dh dw h_out w_out))
  : kernel_desc
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw_l) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw_l) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 et (b*cout*h_out*w_out)).
         (gy |-> sy) **
         pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
                 acc1 sy tid ==
                 convT2d_out_at b cin h_in w_in cout kh kw
                                sh sw ph pw dh dw
                                h_out w_out sx sw_l sbias tid)))
  = [@@inline_let] let nthr : (x : szp { SZ.v x == b * cout * h_out * w_out }) =
      b *^ cout *^ h_out *^ w_out in {
  nthr = nthr;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = convt2d_naive_setup b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                 h_out w_out nthr gx gw gbias gy;
  teardown = convt2d_naive_teardown b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                    h_out w_out nthr gx gw gbias gy;
  kpre  = kpre #et b cin h_in w_in cout kh kw h_out w_out
               #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias sy0 fx fw fb;
  kpost = kpost #et b cin h_in w_in cout kh kw sh sw ph pw dh dw
                h_out w_out
                #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias fx fw fb;
  f = kf b cin h_in w_in cout kh kw sh sw ph pw dh dw h_out w_out
        gx gw gbias gy;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn convt2d_naive_gpu
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
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw_l : chest1 et (cin*cout*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  norewrite
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          convT2d_size_req b cin h_in w_in cout kh kw
                           sh sw ph pw dh dw h_out w_out)
  ensures
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
                              h_out w_out sx sw_l sbias tid))
{
  launch_sync (kdesc b cin h_in w_in cout kh kw sh sw ph pw dh dw
                     h_out w_out gx gw gbias gy)
}
