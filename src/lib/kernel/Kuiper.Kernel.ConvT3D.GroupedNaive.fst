module Kuiper.Kernel.ConvT3D.GroupedNaive

(* Direct grouped ConvTranspose3D implementation.  See [.fsti] for the
   contract.  The kernel computes one output voxel per thread; the
   inner accumulation over the [(ic, kd_i, kh_i, kw_i)] taps is a
   single while-loop matched up to
   [Kuiper.Spec.ConvTranspose3D.__convT3d_single] via the
   [conv1d_partial_at] proof pattern (loop invariant tracks
   [acc == convT3d_partial_at ... k]; step lemma extends by one
   tap).  Setup, teardown, and kpre/kpost sendability are all
   discharged at the [kdesc] level. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv3D
open Kuiper.Spec.ConvTranspose3D
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

let reached_bound (k n bound : nat)
  : Lemma
      (requires k <= bound /\ n == bound /\ not (k < n))
      (ensures k == bound)
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
  (b : pos{b==8}) (cin_pg : pos{cin_pg==8}) (d_in : pos{d_in==12}) (h_in : pos{h_in==24}) (w_in : pos{w_in==48}) (cout_pg : pos{cout_pg==8}) (kd : pos{kd==3}) (kh : pos{kh==5}) (kw : pos{kw==7})
  (d_out : pos{d_out==24}) (h_out : pos{h_out==48}) (w_out : pos{w_out==96})
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in))
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw))
  (#lbias : layout1 (4 * cout_pg))
  (#ly : layout1 (b * (4 * cout_pg) * d_out * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (sy0 : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out))
  (fx fw fb : perm)
  (tid : natlt (b * (4 * cout_pg) * d_out * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sw_l **
    gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> acc1 sy0 tid

unfold
let kpost
  (#et:Type) {| scalar et |}
  (b : pos{b==8}) (cin_pg : pos{cin_pg==8}) (d_in : pos{d_in==12}) (h_in : pos{h_in==24}) (w_in : pos{w_in==48}) (cout_pg : pos{cout_pg==8}) (kd : pos{kd==3}) (kh : pos{kh==5}) (kw : pos{kw==7})
  (sd : pos{sd==2}) (sh : pos{sh==2}) (sw : pos{sw==2}) (pd : nat{pd==1}) (ph : nat{ph==2}) (pw : nat{pw==3}) (dd : pos{dd==1}) (dh : pos{dh==1}) (dw : pos{dw==1})
  (d_out : pos{d_out==24}) (h_out : pos{h_out==48}) (w_out : pos{w_out==96})
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in))
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw))
  (#lbias : layout1 (4 * cout_pg))
  (#ly : layout1 (b * (4 * cout_pg) * d_out * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (fx fw fb : perm)
  (tid : natlt (b * (4 * cout_pg) * d_out * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sw_l **
    gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                                   sd sh sw pd ph pw dd dh dw
                                   d_out h_out w_out sx sw_l sbias tid

#push-options "--z3rlimit 60"

(* Inner-loop helper: read tap from x with strided + zero-padded
   ConvTranspose semantics.  Reads x[bi, ic, num_d/sd, num_h/sh, num_w/sw]
   iff each numerator is non-negative, divisible by stride, and within range. *)
inline_for_extraction noextract
fn read_x_strided_pad_3d
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==8}) (cin_pg : szp{SZ.v cin_pg==8}) (d_in : szp{SZ.v d_in==12}) (h_in : szp{SZ.v h_in==24}) (w_in : szp{SZ.v w_in==48})
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in)) {| ctlayout lx |}
  (gx : array1 et lx)
  (#sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (#fx : perm)
  (bi : szlt b)
  (ic : szlt (4 * cin_pg))
  (od_pd : sz)
  (oh_ph : sz)
  (ow_pw : sz)
  (kd_dd : sz)
  (kh_dh : sz)
  (kw_dw : sz)
  (sd sh sw : szp)
  (#_ : squash (SZ.fits (b * 4 * cin_pg * d_in * h_in * w_in)))
  preserves
    gpu **
    gx |-> Frac fx sx
  returns
    v : et
  ensures
    pure (
      let d_num : int = SZ.v od_pd - SZ.v kd_dd in
      let h_num : int = SZ.v oh_ph - SZ.v kh_dh in
      let w_num : int = SZ.v ow_pw - SZ.v kw_dw in
      v == (if d_num >= 0 && h_num >= 0 && w_num >= 0
              && d_num % SZ.v sd = 0 && h_num % SZ.v sh = 0 && w_num % SZ.v sw = 0
              && d_num / SZ.v sd < d_in
              && h_num / SZ.v sh < h_in
              && w_num / SZ.v sw < w_in
            then acc1 sx
                 ((((bi * (4 * cin_pg) + ic) * d_in + d_num / SZ.v sd) * h_in
                                                              + h_num / SZ.v sh) * w_in
                    + w_num / SZ.v sw)
            else zero))
{
  let cin : szp = 4sz *^ cin_pg;
  if (od_pd >=^ kd_dd && oh_ph >=^ kh_dh && ow_pw >=^ kw_dw) {
    let d_num = od_pd -^ kd_dd;
    let h_num = oh_ph -^ kh_dh;
    let w_num = ow_pw -^ kw_dw;
    let d_rem = d_num %^ sd;
    let h_rem = h_num %^ sh;
    let w_rem = w_num %^ sw;
    if (d_rem = 0sz && h_rem = 0sz && w_rem = 0sz) {
      let di = d_num /^ sd;
      let hi = h_num /^ sh;
      let wi = w_num /^ sw;
      if (di <^ d_in && hi <^ h_in && wi <^ w_in) {
        Math.lemma_mult_lt_right cin bi b;
        Math.lemma_mult_le_right d_in (bi * cin + ic + 1) (b * cin);
        Math.lemma_mult_le_right h_in ((bi * cin + ic) * d_in + di + 1)
                                      (b * cin * d_in);
        Math.lemma_mult_le_right w_in (((bi * cin + ic) * d_in + di) * h_in + hi + 1)
                                      (b * cin * d_in * h_in);
        let p1 = bi *^ cin +^ ic;
        let p2 = p1 *^ d_in +^ di;
        let p3 = p2 *^ h_in +^ hi;
        let flat : szlt (b * (4 * cin_pg) * d_in * h_in * w_in) = p3 *^ w_in +^ wi;
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

#push-options "--z3rlimit 60"

(* Read a weight tap [(ic, oc, kd_i, kh_i, kw_i)] from the flat weight array.
   ConvT layout is (cin, cout, kd, kh, kw). *)
inline_for_extraction noextract
fn read_w_tap_t3
  (#et : Type0) {| scalar et |}
  (cin_pg : szp{SZ.v cin_pg==8}) (cout_pg : szp{SZ.v cout_pg==8}) (kd : szp{SZ.v kd==3}) (kh : szp{SZ.v kh==5}) (kw : szp{SZ.v kw==7})
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw)) {| ctlayout lw |}
  (gw : array1 et lw)
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (#fw : perm)
  (ic : szlt (4 * cin_pg)) (oc : szlt cout_pg)
  (kd_i : szlt kd) (kh_i : szlt kh) (kw_i : szlt kw)
  (#_ : squash (SZ.fits (4 * cin_pg * cout_pg * kd * kh * kw)))
  preserves
    gpu **
    gw |-> Frac fw sw_l
  returns
    v : et
  ensures
    pure (v == acc1 sw_l
              ((((ic * cout_pg + oc) * kd + kd_i) * kh + kh_i) * kw + kw_i))
{
  let cin : szp = 4sz *^ cin_pg;
  Math.lemma_mult_lt_right cout_pg ic cin;
  Math.lemma_mult_le_right kd (ic * cout_pg + oc + 1) (cin * cout_pg);
  Math.lemma_mult_le_right kh ((ic * cout_pg + oc) * kd + kd_i + 1) (cin * cout_pg * kd);
  Math.lemma_mult_le_right kw (((ic * cout_pg + oc) * kd + kd_i) * kh + kh_i + 1)
                              (cin * cout_pg * kd * kh);
  let p1 = ic *^ cout_pg +^ oc;
  let p2 = p1 *^ kd +^ kd_i;
  let p3 = p2 *^ kh +^ kh_i;
  let flat : szlt ((4 * cin_pg) * cout_pg * kd * kh * kw) = p3 *^ kw +^ kw_i;
  tensor_read gw (flat, ())
}

#pop-options

#push-options "--z3rlimit 60 --fuel 2 --ifuel 1"

(* Local helper: partial convT3d sum over the linearised
   (ic, kd, kh, kw) index up to [to], with all parameters explicit.
   Used as the loop-invariant predicate for [kf]'s accumulator. *)
unfold
let convT3d_partial_at
  (#et : Type) {| scalar et |}
  (b : pos{b==8}) (cin_pg : pos{cin_pg==8}) (d_in : pos{d_in==12}) (h_in : pos{h_in==24}) (w_in : pos{w_in==48}) (cout_pg : pos{cout_pg==8})
  (kd : pos{kd==3}) (kh : pos{kh==5}) (kw : pos{kw==7})
  (sd : pos{sd==2}) (sh : pos{sh==2}) (sw : pos{sw==2}) (pd : nat{pd==1}) (ph : nat{ph==2}) (pw : nat{pw==3}) (dd : pos{dd==1}) (dh : pos{dh==1}) (dw : pos{dw==1})
  (d_out : nat{d_out==24}) (h_out : nat{h_out==48}) (w_out : nat{w_out==96})
  (sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (bi : natlt b) (oc : natlt (4 * cout_pg))
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= cin_pg * kd * kh * kw})
  : GTot et
  = let g : natlt 4 = oc / cout_pg in
    let oc_pg : natlt cout_pg = oc % cout_pg in
    __convT3d_single kd kh kw sd sh sw pd ph pw dd dh dw
      (convT3d_group_input
        (lseq_to_t5 b (4 * cin_pg) d_in h_in w_in sx) g)
      (convT3d_group_weight
        (lseq_to_t5 (4 * cin_pg) cout_pg kd kh kw sw_l) g)
      bi oc_pg od oh ow to

(* Step lemma for [convT3d_partial_at]. *)
let convT3d_partial_at_step
  (#et : Type) {| scalar et |}
  (b : pos{b==8}) (cin_pg : pos{cin_pg==8}) (d_in : pos{d_in==12}) (h_in : pos{h_in==24}) (w_in : pos{w_in==48}) (cout_pg : pos{cout_pg==8})
  (kd : pos{kd==3}) (kh : pos{kh==5}) (kw : pos{kw==7})
  (sd : pos{sd==2}) (sh : pos{sh==2}) (sw : pos{sw==2}) (pd : nat{pd==1}) (ph : nat{ph==2}) (pw : nat{pw==3}) (dd : pos{dd==1}) (dh : pos{dh==1}) (dw : pos{dw==1})
  (d_out : nat{d_out==24}) (h_out : nat{h_out==48}) (w_out : nat{w_out==96})
  (sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (bi : natlt b) (oc : natlt (4 * cout_pg))
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin_pg * kd * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank3_ic cin_pg kd kh kw i in
      let kd_i = unrank3_kd cin_pg kd kh kw i in
      let kh_i = unrank3_kh cin_pg kd kh kw i in
      let kw_i = unrank3_kw cin_pg kd kh kw i in
      let d_num : int = od + pd - kd_i * dd in
      let h_num : int = oh + ph - kh_i * dh in
      let w_num : int = ow + pw - kw_i * dw in
      convT3d_partial_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                         sd sh sw pd ph pw dd dh dw
                         d_out h_out w_out sx sw_l sbias bi oc od oh ow to ==
      add (convT3d_partial_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias bi oc od oh ow
                              (to - 1))
          (mul (read_strided_padded_3d
                  (convT3d_group_input
                    (lseq_to_t5 b (4 * cin_pg) d_in h_in w_in sx)
                    (oc / cout_pg)) bi ic
                  sd sh sw d_num h_num w_num)
               (t5acc (convT3d_group_weight
                        (lseq_to_t5 (4 * cin_pg) cout_pg kd kh kw sw_l)
                        (oc / cout_pg))
                      ic (oc % cout_pg) kd_i kh_i kw_i))))
  = __convT3d_single_lemma cin_pg kd kh kw sd sh sw pd ph pw dd dh dw
      (convT3d_group_input
        (lseq_to_t5 b (4 * cin_pg) d_in h_in w_in sx) (oc / cout_pg))
      (convT3d_group_weight
        (lseq_to_t5 (4 * cin_pg) cout_pg kd kh kw sw_l) (oc / cout_pg))
      bi (oc % cout_pg) od oh ow to

(* Per-thread ConvT body: decode tid, run the inner accumulator loop,
   add bias, write to the output cell.  Spec-connection is now
   discharged via the [convT3d_partial_at] loop invariant; setup,
   teardown, and sendability are discharged at the [kdesc] level. *)
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==8}) (cin_pg : szp{SZ.v cin_pg==8}) (d_in : szp{SZ.v d_in==12}) (h_in : szp{SZ.v h_in==24}) (w_in : szp{SZ.v w_in==48}) (cout_pg : szp{SZ.v cout_pg==8})
  (kd : szp{SZ.v kd==3}) (kh : szp{SZ.v kh==5}) (kw : szp{SZ.v kw==7})
  (sd : szp{SZ.v sd==2}) (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==2}) (pd : sz{SZ.v pd==1}) (ph : sz{SZ.v ph==2}) (pw : sz{SZ.v pw==3}) (dd : szp{SZ.v dd==1}) (dh : szp{SZ.v dh==1}) (dw : szp{SZ.v dw==1})
  (d_out : szp{SZ.v d_out==24}) (h_out : szp{SZ.v h_out==48}) (w_out : szp{SZ.v w_out==96})
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 (4 * cout_pg)) {| ctlayout lbias |}
  (#ly : layout1 (b * (4 * cout_pg) * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (b * (4 * cout_pg) * d_out * h_out * w_out > 0))
  (#_ : squash (SZ.fits (cin_pg * kd * kh * kw) /\
                SZ.fits (kd * kh * kw) /\
                SZ.fits (kh * kw) /\
                SZ.fits (h_out * w_out) /\
                SZ.fits (d_out * h_out * w_out) /\
                SZ.fits (4 * cout_pg * d_out * h_out * w_out) /\
                SZ.fits (b * (4 * cin_pg) * d_in * h_in * w_in) /\
                SZ.fits ((4 * cin_pg) * cout_pg * kd * kh * kw) /\
                SZ.fits (d_out + pd) /\
                SZ.fits (h_out + ph) /\
                SZ.fits (w_out + pw) /\
                SZ.fits (kd * dd) /\
                SZ.fits (kh * dh) /\
                SZ.fits (kw * dw)))
  (tid : szlt (b * (4 * cout_pg) * d_out * h_out * w_out))
  ()
  norewrite
  preserves gpu
  requires
    kpre #et b cin_pg d_in h_in w_in cout_pg kd kh kw d_out h_out w_out
         #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias sy0 fx fw fb tid
  ensures
    kpost #et b cin_pg d_in h_in w_in cout_pg kd kh kw sd sh sw pd ph pw dd dh dw
          d_out h_out w_out
          #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias fx fw fb tid
{
  let cin : szp = 4sz *^ cin_pg;
  let cout : szp = 4sz *^ cout_pg;
  let how : sz = h_out *^ w_out;
  named_mul_value h_out w_out how;
  let dhw : sz = d_out *^ how;
  named_mul_value d_out how dhw;
  let cdhw : sz = cout *^ dhw;
  named_mul_value cout dhw cdhw;
  output_decode_facts b cout d_out h_out
    w_out tid how dhw cdhw;
  div_lt_product tid b cdhw;
  let bi : szlt b = tid /^ cdhw;
  let r1 : szlt cdhw = tid %^ cdhw;
  let oc : szlt cout = r1 /^ dhw;
  let g : szlt 4 = oc /^ cout_pg;
  let oc_pg : szlt cout_pg = oc %^ cout_pg;
  let r2 : szlt dhw = r1 %^ dhw;
  let od : szlt d_out = r2 /^ how;
  let r3 : szlt how = r2 %^ how;
  let oh : szlt h_out = r3 /^ w_out;
  let ow : szlt w_out = r3 %^ w_out;

  let kh_kw : sz = kh *^ kw;
  named_mul_value kh kw kh_kw;
  let kd_kh_kw : sz = kd *^ kh_kw;
  named_mul_value kd kh_kw kd_kh_kw;
  flattened_taps_fit cin_pg kd kh kw kh_kw kd_kh_kw;
  let n_taps : sz = cin_pg *^ kd_kh_kw;
  named_mul_value cin_pg kd_kh_kw n_taps;
  flatten_taps cin_pg kd kh kw kh_kw kd_kh_kw n_taps;

  let od_pd : sz = od +^ pd;
  let oh_ph : sz = oh +^ ph;
  let ow_pw : sz = ow +^ pw;

  let mut acc : et = zero;
  let mut k : sz = 0sz;

  while (!k <^ n_taps)
    invariant
      exists* (vk : sz{SZ.v vk <= cin_pg * kd * kh * kw}).
        k |-> vk **
        acc |-> convT3d_partial_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                  sd sh sw pd ph pw dd dh dw
                  d_out h_out w_out sx sw_l sbias bi oc od oh ow vk
    invariant pure (SZ.fits (cin_pg * kd * kh * kw))
    invariant gx |-> Frac (fx /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sx
    invariant gw |-> Frac (fw /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sw_l
    invariant gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias
    invariant gpu
    decreases (cin_pg * kd * kh * kw - SZ.v !k)
  {
    let kk = !k;
    assert pure (SZ.v kk < cin_pg * SZ.v kd_kh_kw);
    div_lt_product kk cin_pg kd_kh_kw;
    let ic_pg : szlt cin_pg = kk /^ kd_kh_kw;
    let ic : szlt cin = g *^ cin_pg +^ ic_pg;
    let r  : szlt kd_kh_kw = kk %^ kd_kh_kw;
    assert pure (SZ.v r < kd * SZ.v kh_kw);
    div_lt_product r kd kh_kw;
    let kd_i : szlt kd = r /^ kh_kw;
    let r2  : szlt kh_kw = r %^ kh_kw;
    assert pure (SZ.v r2 < kh * kw);
    div_lt_product r2 kh kw;
    let kh_i : szlt kh = r2 /^ kw;
    let kw_i : szlt kw = r2 %^ kw;

    assert pure (SZ.v kk < cin_pg * kd * kh * kw);
    assert pure (SZ.v ic_pg == SZ.v kk / SZ.v kd_kh_kw);
    assert pure (SZ.v r == SZ.v kk % SZ.v kd_kh_kw);
    assert pure (SZ.v kd_i == SZ.v r / SZ.v kh_kw);
    assert pure (SZ.v r2 == SZ.v r % SZ.v kh_kw);
    assert pure (SZ.v kh_i == SZ.v r2 / kw);
    assert pure (SZ.v kw_i == SZ.v r2 % kw);
    unrank3_from_steps cin_pg kd kh kw kk kh_kw
      kd_kh_kw n_taps ic_pg r kd_i
      r2 kh_i kw_i;

    let kd_dd : sz = kd_i *^ dd;
    let kh_dh : sz = kh_i *^ dh;
    let kw_dw : sz = kw_i *^ dw;
    let xv = read_x_strided_pad_3d b cin_pg d_in h_in w_in gx bi ic
                                   od_pd oh_ph ow_pw kd_dd kh_dh kw_dw sd sh sw;
    let wv = read_w_tap_t3 cin_pg cout_pg kd kh kw
      gw ic oc_pg kd_i kh_i kw_i;
    let prod = mul xv wv;
    let acc0 = !acc;
    (* Establish the step equation: prod equals the lemma's per-tap product. *)
    assert pure (xv == read_strided_padded_3d
                         (convT3d_group_input
                           (lseq_to_t5 b (4 * cin_pg) d_in h_in w_in sx) g)
                         bi ic_pg
                         sd sh sw
                         (od + pd - SZ.v kd_i * dd)
                         (oh + ph - SZ.v kh_i * dh)
                         (ow + pw - SZ.v kw_i * dw));
    assert pure (wv == t5acc
      (convT3d_group_weight
        (lseq_to_t5 (4 * cin_pg) cout_pg kd kh kw sw_l) g)
      ic_pg oc_pg kd_i kh_i kw_i);
    convT3d_partial_at_step b cin_pg d_in h_in w_in cout_pg kd kh kw
      sd sh sw pd ph pw dd dh dw
      d_out h_out w_out sx sw_l sbias bi oc od oh ow (SZ.v kk + 1);
    acc := add acc0 prod;
    assert pure (SZ.v kk < cin_pg * kd * kh * kw);
    decreases_after_increment (cin_pg * kd * kh * kw) kk;
    let knew : sz = !k +^ 1sz;
    assert pure (SZ.v knew == SZ.v kk + 1);
    assert pure (SZ.v knew <= cin_pg * kd * kh * kw);
    k := knew;
  };

  (* Loop exit: !k = vk = n_taps = cin*kd*kh*kw, so acc holds the full
     partial sum [__convT3d_single ... (cin*kd*kh*kw)] (via the [unfold]
     [convT3d_partial_at]).  Adding the bias gives [convT3d_single], which
     equals [convT3d_grouped_out_at] applied to [tid] once the kernel-side decode
     of [tid] is shown to match the spec-side decode (modulo associativity
     of [*]).  Same discharge pattern as [Kuiper.Kernel.Conv3D.Naive.kf]. *)
  let k_done = !k;
  reached_bound k_done n_taps (cin_pg * kd * kh * kw);
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

  assert pure (result == convT3d_grouped_out_at b cin_pg d_in h_in w_in
                                        cout_pg kd kh kw
                                        sd sh sw
                                        pd ph pw
                                        dd dh dw
                                        d_out h_out w_out
                                        sx sw_l sbias tid);
  tensor_write_cell gy (tid, ()) result
}

#pop-options

#push-options "--z3rlimit 60"

(* Ghost setup: factor the launcher's full-permission frame into N per-thread
   slices.  Mirrors [Kuiper.Kernel.Conv1D.Naive.conv1d_naive_setup]. *)
ghost
fn convt3d_grouped_naive_setup
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==8}) (cin_pg : szp{SZ.v cin_pg==8}) (d_in : szp{SZ.v d_in==12}) (h_in : szp{SZ.v h_in==24}) (w_in : szp{SZ.v w_in==48}) (cout_pg : szp{SZ.v cout_pg==8}) (kd : szp{SZ.v kd==3}) (kh : szp{SZ.v kh==5}) (kw : szp{SZ.v kw==7})
  (sd : szp{SZ.v sd==2}) (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==2}) (pd : sz{SZ.v pd==1}) (ph : sz{SZ.v ph==2}) (pw : sz{SZ.v pw==3}) (dd : szp{SZ.v dd==1}) (dh : szp{SZ.v dh==1}) (dw : szp{SZ.v dw==1})
  (d_out : szp{SZ.v d_out==24}) (h_out : szp{SZ.v h_out==48}) (w_out : szp{SZ.v w_out==96})
  (nthr : szp { SZ.v nthr == b * (4 * cout_pg) * d_out * h_out * w_out })
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in))
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw))
  (#lbias : layout1 (4 * cout_pg))
  (#ly : layout1 (b * (4 * cout_pg) * d_out * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (convT3d_grouped_size_req b cin_pg d_in h_in w_in cout_pg kd kh kw
                                 sd sh sw pd ph pw dd dh dw
                                 d_out h_out w_out))
  ()
  norewrite
  requires
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw_l) **
    (gbias |-> Frac fb sbias) **
    (gy |-> sy0)
  ensures
    (forall+ (tid : natlt nthr).
       kpre #et b cin_pg d_in h_in w_in cout_pg kd kh kw d_out h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw_l sbias sy0 fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
{
  tensor_pts_to_ref gy;
  tensor_share_n gx (b * (4 * cout_pg) * d_out * h_out * w_out);
  tensor_share_n gw (b * (4 * cout_pg) * d_out * h_out * w_out);
  tensor_share_n gbias (b * (4 * cout_pg) * d_out * h_out * w_out);
  tensor_explode gy;
  forevery_iso (abs_bij #(b * (4 * cout_pg) * d_out * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * (4 * cout_pg) * d_out * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc sy0 i);
  forevery_ext
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc sy0 (abs_bij.gg i))
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias)
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sw_l)
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_zip
    (fun (_ : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sx)
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sw_l) **
       (gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (b * (4 * cout_pg) * d_out * h_out * w_out) nthr;
  ()
}

(* Ghost teardown: gather N per-thread slices back into the launcher
   postcondition.  Symmetric inverse of [convt3d_grouped_naive_setup]. *)
ghost
fn convt3d_grouped_naive_teardown
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==8}) (cin_pg : szp{SZ.v cin_pg==8}) (d_in : szp{SZ.v d_in==12}) (h_in : szp{SZ.v h_in==24}) (w_in : szp{SZ.v w_in==48}) (cout_pg : szp{SZ.v cout_pg==8}) (kd : szp{SZ.v kd==3}) (kh : szp{SZ.v kh==5}) (kw : szp{SZ.v kw==7})
  (sd : szp{SZ.v sd==2}) (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==2}) (pd : sz{SZ.v pd==1}) (ph : sz{SZ.v ph==2}) (pw : sz{SZ.v pw==3}) (dd : szp{SZ.v dd==1}) (dh : szp{SZ.v dh==1}) (dw : szp{SZ.v dw==1})
  (d_out : szp{SZ.v d_out==24}) (h_out : szp{SZ.v h_out==48}) (w_out : szp{SZ.v w_out==96})
  (nthr : szp { SZ.v nthr == b * (4 * cout_pg) * d_out * h_out * w_out })
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in))
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw))
  (#lbias : layout1 (4 * cout_pg))
  (#ly : layout1 (b * (4 * cout_pg) * d_out * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#fx #fw #fb : perm)
  (#_ : squash (convT3d_grouped_size_req b cin_pg d_in h_in w_in cout_pg kd kh kw
                                 sd sh sw pd ph pw dd dh dw
                                 d_out h_out w_out))
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpost #et b cin_pg d_in h_in w_in cout_pg kd kh kw
             sd sh sw pd ph pw dd dh dw
             d_out h_out w_out
             #lx #lw #lbias #ly
             gx gw gbias gy sx sw_l sbias fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw_l) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < b*(4*cout_pg)*d_out*h_out*w_out}).
               acc1 sy tid ==
               convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias tid))
{
  forevery_rw_size nthr (b * (4 * cout_pg) * d_out * h_out * w_out)
    #(kpost #et b cin_pg d_in h_in w_in cout_pg kd kh kw
            sd sh sw pd ph pw dd dh dw
            d_out h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw_l sbias fx fw fb);
  forevery_unzip
    (fun (_ : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sx)
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sw_l) **
       (gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                                     sd sh sw pd ph pw dd dh dw
                                     d_out h_out w_out sx sw_l sbias i));
  forevery_unzip
    (fun (_ : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sw_l)
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                                     sd sh sw pd ph pw dd dh dw
                                     d_out h_out w_out sx sw_l sbias i));
  forevery_unzip
    (fun (_ : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * (4 * cout_pg) * d_out * h_out * w_out)) sbias)
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       Cell gy (idx1 i) |-> convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                                    sd sh sw pd ph pw dd dh dw
                                    d_out h_out w_out sx sw_l sbias i);
  tensor_gather_n gx (b * (4 * cout_pg) * d_out * h_out * w_out);
  tensor_gather_n gw (b * (4 * cout_pg) * d_out * h_out * w_out);
  tensor_gather_n gbias (b * (4 * cout_pg) * d_out * h_out * w_out);
  let sy : chest1 et (b * (4 * cout_pg) * d_out * h_out * w_out) =
    hide (mk1
            (fun (tid : nat{tid < b * (4 * cout_pg) * d_out * h_out * w_out}) ->
               convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias tid));
  forevery_ext
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       Cell gy (idx1 i) |-> convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                                    sd sh sw pd ph pw dd dh dw
                                    d_out h_out w_out sx sw_l sbias i)
    (fun (i : natlt (b * (4 * cout_pg) * d_out * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc (reveal sy) (abs_bij.gg i));
  forevery_iso_back (abs_bij #(b * (4 * cout_pg) * d_out * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * (4 * cout_pg) * d_out * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc (reveal sy) i);
  tensor_implode gy;
  ()
}

#pop-options

#push-options "--z3rlimit 60 --fuel 2 --ifuel 2"

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==8}) (cin_pg : szp{SZ.v cin_pg==8}) (d_in : szp{SZ.v d_in==12}) (h_in : szp{SZ.v h_in==24}) (w_in : szp{SZ.v w_in==48}) (cout_pg : szp{SZ.v cout_pg==8})
  (kd : szp{SZ.v kd==3}) (kh : szp{SZ.v kh==5}) (kw : szp{SZ.v kw==7})
  (sd : szp{SZ.v sd==2}) (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==2}) (pd : sz{SZ.v pd==1}) (ph : sz{SZ.v ph==2}) (pw : sz{SZ.v pw==3}) (dd : szp{SZ.v dd==1}) (dh : szp{SZ.v dh==1}) (dw : szp{SZ.v dw==1})
  (d_out : szp{SZ.v d_out==24}) (h_out : szp{SZ.v h_out==48}) (w_out : szp{SZ.v w_out==96})
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 (4 * cout_pg)) {| ctlayout lbias |}
  (#ly : layout1 (b * (4 * cout_pg) * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (is_global gx /\ is_global gw /\
                is_global gbias /\ is_global gy /\
                convT3d_grouped_size_req b cin_pg d_in h_in w_in cout_pg kd kh kw
                                 sd sh sw pd ph pw dd dh dw
                                 d_out h_out w_out))
  : kernel_desc
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw_l) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw_l) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out)).
         (gy |-> sy) **
         pure (forall (tid : nat{tid < b*(4*cout_pg)*d_out*h_out*w_out}).
                 acc1 sy tid ==
                 convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                                sd sh sw pd ph pw dd dh dw
                                d_out h_out w_out sx sw_l sbias tid)))
  = [@@inline_let] let nthr : (x : szp { SZ.v x == b * (4 * cout_pg) * d_out * h_out * w_out }) =
      b *^ (4sz *^ cout_pg) *^ d_out *^ h_out *^ w_out in {
  nthr = nthr;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = convt3d_grouped_naive_setup b cin_pg d_in h_in w_in cout_pg kd kh kw
                                 sd sh sw pd ph pw dd dh dw
                                 d_out h_out w_out nthr gx gw gbias gy;
  teardown = convt3d_grouped_naive_teardown b cin_pg d_in h_in w_in cout_pg kd kh kw
                                    sd sh sw pd ph pw dd dh dw
                                    d_out h_out w_out nthr gx gw gbias gy;
  kpre  = kpre #et b cin_pg d_in h_in w_in cout_pg kd kh kw d_out h_out w_out
               #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias sy0 fx fw fb;
  kpost = kpost #et b cin_pg d_in h_in w_in cout_pg kd kh kw sd sh sw pd ph pw
                dd dh dw d_out h_out w_out
                #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias fx fw fb;
  f = kf b cin_pg d_in h_in w_in cout_pg kd kh kw sd sh sw pd ph pw dd dh dw
        d_out h_out w_out gx gw gbias gy;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn convt3d_grouped_naive_gpu
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==8}) (cin_pg : szp{SZ.v cin_pg==8}) (d_in : szp{SZ.v d_in==12}) (h_in : szp{SZ.v h_in==24}) (w_in : szp{SZ.v w_in==48}) (cout_pg : szp{SZ.v cout_pg==8})
  (kd : szp{SZ.v kd==3}) (kh : szp{SZ.v kh==5}) (kw : szp{SZ.v kw==7})
  (sd : szp{SZ.v sd==2}) (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==2}) (pd : sz{SZ.v pd==1}) (ph : sz{SZ.v ph==2}) (pw : sz{SZ.v pw==3}) (dd : szp{SZ.v dd==1}) (dh : szp{SZ.v dh==1}) (dw : szp{SZ.v dw==1})
  (d_out : szp{SZ.v d_out==24}) (h_out : szp{SZ.v h_out==48}) (w_out : szp{SZ.v w_out==96})
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 (4 * cout_pg)) {| ctlayout lbias |}
  (#ly : layout1 (b * (4 * cout_pg) * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out))
  (#fx #fw #fb : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          convT3d_grouped_size_req b cin_pg d_in h_in w_in cout_pg kd kh kw
                           sd sh sw pd ph pw dd dh dw
                           d_out h_out w_out)
  ensures
    (exists* (sy : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*(4*cout_pg)*d_out*h_out*w_out}).
               acc1 sy tid ==
               convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias tid))
{
  launch_sync (kdesc b cin_pg d_in h_in w_in cout_pg kd kh kw sd sh sw pd ph pw
                     dd dh dw d_out h_out w_out gx gw gbias gy)
}
