module Kuiper.Kernel.ConvT2D.GroupedNaive

(* Direct grouped ConvTranspose2D implementation.  See [.fsti] for the
   contract.  The kernel computes one output pixel per thread; the
   inner accumulation over the [(ic, kh_i, kw_i)] taps is a single
   while-loop matched up to [Kuiper.Spec.ConvTranspose2D.__convT2d_single]
   via the [conv1d_partial_at] proof pattern (loop invariant tracks
   [acc == convT2d_partial_at ... k]; step lemma extends by one
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
  (b : pos{b==16}) (cin_pg : pos{cin_pg==8}) (h_in : pos{h_in==128}) (w_in : pos{w_in==256}) (cout_pg : pos{cout_pg==16}) (kh : pos{kh==3}) (kw : pos{kw==5})
  (h_out : pos{h_out==257}) (w_out : pos{w_out==766})
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in))
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw))
  (#lbias : layout1 (4 * cout_pg))
  (#ly : layout1 (b * (4 * cout_pg) * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (sy0 : chest1 et (b*(4*cout_pg)*h_out*w_out))
  (fx fw fb : perm)
  (tid : natlt (b * (4 * cout_pg) * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * (4 * cout_pg) * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * (4 * cout_pg) * h_out * w_out)) sw_l **
    gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> acc1 sy0 tid

unfold
let kpost
  (#et:Type) {| scalar et |}
  (b : pos{b==16}) (cin_pg : pos{cin_pg==8}) (h_in : pos{h_in==128}) (w_in : pos{w_in==256}) (cout_pg : pos{cout_pg==16}) (kh : pos{kh==3}) (kw : pos{kw==5})
  (sh : pos{sh==2}) (sw : pos{sw==3}) (ph : nat{ph==1}) (pw : nat{pw==2}) (dh : pos{dh==2}) (dw : pos{dw==1})
  (h_out : pos{h_out==257}) (w_out : pos{w_out==766})
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in))
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw))
  (#lbias : layout1 (4 * cout_pg))
  (#ly : layout1 (b * (4 * cout_pg) * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (fx fw fb : perm)
  (tid : natlt (b * (4 * cout_pg) * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * (4 * cout_pg) * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * (4 * cout_pg) * h_out * w_out)) sw_l **
    gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                                   sh sw ph pw dh dw
                                   h_out w_out sx sw_l sbias tid

#push-options "--z3rlimit 60"

(* Inner-loop helper: read tap from x with strided + zero-padded
   ConvTranspose semantics.  Reads x[bi, ic, num_h/sh, num_w/sw] iff
   num_h, num_w >= 0, divisible by sh/sw, and within range. *)
inline_for_extraction noextract
fn read_x_strided_pad
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==16}) (cin_pg : szp{SZ.v cin_pg==8}) (h_in : szp{SZ.v h_in==128}) (w_in : szp{SZ.v w_in==256})
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in)) {| ctlayout lx |}
  (gx : array1 et lx)
  (#sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (#fx : perm)
  (bi : szlt b)
  (ic : szlt (4 * cin_pg))
  (oh_ph : sz)   (* oh + ph *)
  (ow_pw : sz)   (* ow + pw *)
  (kh_dh : sz)   (* kh_i * dh *)
  (kw_dw : sz)   (* kw_i * dw *)
  (sh sw : szp)
  (#_ : squash (SZ.fits (b * 4 * cin_pg * h_in * w_in)))
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
                 (((bi * (4 * cin_pg) + ic) * h_in + h_num / SZ.v sh) * w_in
                    + w_num / SZ.v sw)
            else zero))
{
  let cin : szp = 4sz *^ cin_pg;
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
        let flat : szlt (b * (4 * cin_pg) * h_in * w_in) = p2 *^ w_in +^ wi;
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

(* Read a weight tap [(ic, oc, kh_i, kw_i)] from the flat weight array.
   Note: ConvT layout is (cin, cout, kh, kw) — different from forward conv. *)
inline_for_extraction noextract
fn read_w_tap_t
  (#et : Type0) {| scalar et |}
  (cin_pg : szp{SZ.v cin_pg==8}) (cout_pg : szp{SZ.v cout_pg==16}) (kh : szp{SZ.v kh==3}) (kw : szp{SZ.v kw==5})
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw)) {| ctlayout lw |}
  (gw : array1 et lw)
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (#fw : perm)
  (ic : szlt (4 * cin_pg)) (oc : szlt cout_pg)
  (kh_i : szlt kh) (kw_i : szlt kw)
  (#_ : squash (SZ.fits (4 * cin_pg * cout_pg * kh * kw)))
  preserves
    gpu **
    gw |-> Frac fw sw_l
  returns
    v : et
  ensures
    pure (v == acc1 sw_l (((ic * cout_pg + oc) * kh + kh_i) * kw + kw_i))
{
  let cin : szp = 4sz *^ cin_pg;
  Math.lemma_mult_lt_right cout_pg ic cin;
  Math.lemma_mult_le_right kh (ic * cout_pg + oc + 1) (cin * cout_pg);
  Math.lemma_mult_le_right kw ((ic * cout_pg + oc) * kh + kh_i + 1) (cin * cout_pg * kh);
  let p1 = ic *^ cout_pg +^ oc;
  let p2 = p1 *^ kh +^ kh_i;
  let flat : szlt ((4 * cin_pg) * cout_pg * kh * kw) = p2 *^ kw +^ kw_i;
  tensor_read gw (flat, ())
}

#pop-options

#push-options "--z3rlimit 60 --fuel 2 --ifuel 1"

(* Local helper: partial convT2d sum over the linearised (ic, kh, kw)
   index up to [to], with all parameters explicit.  Used as the loop-
   invariant predicate for [kf]'s accumulator.  Wrapping the val-only
   [__convT2d_single] avoids unification difficulties when the spec
   appears inside a Pulse [exists*] slprop. *)
unfold
let convT2d_partial_at
  (#et : Type) {| scalar et |}
  (b : pos{b==16}) (cin_pg : pos{cin_pg==8}) (h_in : pos{h_in==128}) (w_in : pos{w_in==256}) (cout_pg : pos{cout_pg==16})
  (kh : pos{kh==3}) (kw : pos{kw==5})
  (sh : pos{sh==2}) (sw : pos{sw==3}) (ph : nat{ph==1}) (pw : nat{pw==2}) (dh : pos{dh==2}) (dw : pos{dw==1})
  (h_out : nat{h_out==257}) (w_out : nat{w_out==766})
  (sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (bi : natlt b) (oc : natlt (4 * cout_pg))
  (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= cin_pg * kh * kw})
  : GTot et
  = let g : natlt 4 = oc / cout_pg in
    let oc_pg : natlt cout_pg = oc % cout_pg in
    __convT2d_single kh kw sh sw ph pw dh dw
      (convT2d_group_input (lseq_to_t4 b (4 * cin_pg) h_in w_in sx) g)
      (convT2d_group_weight
        (lseq_to_t4 (4 * cin_pg) cout_pg kh kw sw_l) g)
      bi oc_pg oh ow to

(* Step lemma for [convT2d_partial_at]: extends the partial sum by one tap. *)
let convT2d_partial_at_step
  (#et : Type) {| scalar et |}
  (b : pos{b==16}) (cin_pg : pos{cin_pg==8}) (h_in : pos{h_in==128}) (w_in : pos{w_in==256}) (cout_pg : pos{cout_pg==16})
  (kh : pos{kh==3}) (kw : pos{kw==5})
  (sh : pos{sh==2}) (sw : pos{sw==3}) (ph : nat{ph==1}) (pw : nat{pw==2}) (dh : pos{dh==2}) (dw : pos{dw==1})
  (h_out : nat{h_out==257}) (w_out : nat{w_out==766})
  (sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (bi : natlt b) (oc : natlt (4 * cout_pg))
  (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin_pg * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank_ic cin_pg kh kw i in
      let kh_i = unrank_kh cin_pg kh kw i in
      let kw_i = unrank_kw cin_pg kh kw i in
      let h_num : int = oh + ph - kh_i * dh in
      let w_num : int = ow + pw - kw_i * dw in
      convT2d_partial_at b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
                         h_out w_out sx sw_l sbias bi oc oh ow to ==
      add (convT2d_partial_at b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias bi oc oh ow (to - 1))
          (mul (read_strided_padded_2d
                  (convT2d_group_input
                    (lseq_to_t4 b (4 * cin_pg) h_in w_in sx)
                    (oc / cout_pg))
                  bi ic sh sw h_num w_num)
               (tacc (convT2d_group_weight
                        (lseq_to_t4 (4 * cin_pg) cout_pg kh kw sw_l)
                        (oc / cout_pg))
                      ic (oc % cout_pg) kh_i kw_i))))
  = __convT2d_single_lemma cin_pg kh kw sh sw ph pw dh dw
      (convT2d_group_input
        (lseq_to_t4 b (4 * cin_pg) h_in w_in sx) (oc / cout_pg))
      (convT2d_group_weight
        (lseq_to_t4 (4 * cin_pg) cout_pg kh kw sw_l) (oc / cout_pg))
      bi (oc % cout_pg) oh ow to

(* Per-thread ConvT body: decode tid, run inner accumulator loop,
   add bias, write output cell.  Spec-connection is now discharged via
   the [convT2d_partial_at] loop invariant; setup, teardown, and
   sendability are discharged at the [kdesc] level (see below). *)
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==16}) (cin_pg : szp{SZ.v cin_pg==8}) (h_in : szp{SZ.v h_in==128}) (w_in : szp{SZ.v w_in==256}) (cout_pg : szp{SZ.v cout_pg==16})
  (kh : szp{SZ.v kh==3}) (kw : szp{SZ.v kw==5})
  (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==3}) (ph : sz{SZ.v ph==1}) (pw : sz{SZ.v pw==2}) (dh : szp{SZ.v dh==2}) (dw : szp{SZ.v dw==1})
  (h_out : szp{SZ.v h_out==257}) (w_out : szp{SZ.v w_out==766})
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 (4 * cout_pg)) {| ctlayout lbias |}
  (#ly : layout1 (b * (4 * cout_pg) * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (b * (4 * cout_pg) * h_out * w_out > 0))
  (#_ : squash (SZ.fits (cin_pg * kh * kw) /\
                SZ.fits (kh * kw) /\
                SZ.fits (h_out * w_out) /\
                SZ.fits (4 * cout_pg * h_out * w_out) /\
                SZ.fits (b * (4 * cin_pg) * h_in * w_in) /\
                SZ.fits ((4 * cin_pg) * cout_pg * kh * kw) /\
                SZ.fits (h_out + ph) /\
                SZ.fits (w_out + pw) /\
                SZ.fits (kh * dh) /\
                SZ.fits (kw * dw)))
  (tid : szlt (b * (4 * cout_pg) * h_out * w_out))
  ()
  norewrite
  preserves gpu
  requires
    kpre #et b cin_pg h_in w_in cout_pg kh kw h_out w_out
         #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias sy0 fx fw fb tid
  ensures
    kpost #et b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
          h_out w_out
          #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias fx fw fb tid
{
  let cin : szp = 4sz *^ cin_pg;
  let cout : szp = 4sz *^ cout_pg;
  let how : sz = h_out *^ w_out;
  let chow : sz = cout *^ how;
  let bi : szlt b = tid /^ chow;
  let r1 : szlt chow = tid %^ chow;
  let oc : szlt cout = r1 /^ how;
  let g : szlt 4 = oc /^ cout_pg;
  let oc_pg : szlt cout_pg = oc %^ cout_pg;
  let r2 : szlt how = r1 %^ how;
  let oh : szlt h_out = r2 /^ w_out;
  let ow : szlt w_out = r2 %^ w_out;

  let kh_kw : sz = kh *^ kw;
  let n_taps : sz = cin_pg *^ kh_kw;

  let oh_ph : sz = oh +^ ph;
  let ow_pw : sz = ow +^ pw;

  let mut acc : et = zero;
  let mut k : sz = 0sz;

  while (!k <^ n_taps)
    invariant
      exists* (vk : sz{SZ.v vk <= cin_pg * kh * kw}).
        k |-> vk **
        acc |-> convT2d_partial_at b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
                  h_out w_out sx sw_l sbias bi oc oh ow vk
    invariant pure (SZ.fits (cin_pg * kh * kw))
    invariant gx |-> Frac (fx /. (b * (4 * cout_pg) * h_out * w_out)) sx
    invariant gw |-> Frac (fw /. (b * (4 * cout_pg) * h_out * w_out)) sw_l
    invariant gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias
    invariant gpu
    decreases (cin_pg * kh * kw - SZ.v !k)
  {
    let kk = !k;
    let ic_pg : szlt cin_pg = kk /^ kh_kw;
    let ic : szlt cin = g *^ cin_pg +^ ic_pg;
    let r : szlt kh_kw = kk %^ kh_kw;
    let kh_i : szlt kh = r /^ kw;
    let kw_i : szlt kw = r %^ kw;

    let kh_dh : sz = kh_i *^ dh;
    let kw_dw : sz = kw_i *^ dw;
    let xv = read_x_strided_pad b cin_pg h_in w_in gx bi ic
                                 oh_ph ow_pw kh_dh kw_dw sh sw;
    let wv = read_w_tap_t cin_pg cout_pg kh kw gw ic oc_pg kh_i kw_i;
    let prod = mul xv wv;
    let acc0 = !acc;
    (* Establish the step equation: prod equals the lemma's per-tap product. *)
    Math.paren_mul_right cin_pg kh kw;
    assert pure (SZ.v n_taps == cin_pg * kh * kw);
    assert pure (SZ.v ic_pg == unrank_ic cin_pg kh kw kk);
    assert pure (SZ.v kh_i == unrank_kh cin_pg kh kw kk);
    assert pure (SZ.v kw_i == unrank_kw cin_pg kh kw kk);
    assert pure (xv == read_strided_padded_2d
                         (convT2d_group_input
                           (lseq_to_t4 b (4 * cin_pg) h_in w_in sx) g)
                         bi ic_pg sh sw
                         (oh + ph - SZ.v kh_i * dh)
                         (ow + pw - SZ.v kw_i * dw));
    assert pure (wv == tacc
      (convT2d_group_weight
        (lseq_to_t4 (4 * cin_pg) cout_pg kh kw sw_l) g)
      ic_pg oc_pg kh_i kw_i);
    convT2d_partial_at_step b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
      h_out w_out sx sw_l sbias bi oc oh ow (SZ.v kk + 1);
    acc := add acc0 prod;
    let knew : sz = !k +^ 1sz;
    assert pure (SZ.v knew <= cin_pg * kh * kw);
    k := knew;
  };

  let bias_v = tensor_read gbias (oc, ());
  let result = add bias_v !acc;
  assert pure (result ==
    convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
      sh sw ph pw dh dw h_out w_out sx sw_l sbias tid);
  tensor_write_cell gy (tid, ()) result
}

#pop-options

#push-options "--z3rlimit 60"

(* Ghost setup: factor the launcher's full-permission frame into N per-thread
   slices.  Mirrors [Kuiper.Kernel.Conv1D.Naive.conv1d_naive_setup]. *)
ghost
fn convt2d_grouped_naive_setup
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==16}) (cin_pg : szp{SZ.v cin_pg==8}) (h_in : szp{SZ.v h_in==128}) (w_in : szp{SZ.v w_in==256}) (cout_pg : szp{SZ.v cout_pg==16}) (kh : szp{SZ.v kh==3}) (kw : szp{SZ.v kw==5})
  (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==3}) (ph : sz{SZ.v ph==1}) (pw : sz{SZ.v pw==2}) (dh : szp{SZ.v dh==2}) (dw : szp{SZ.v dw==1})
  (h_out : szp{SZ.v h_out==257}) (w_out : szp{SZ.v w_out==766})
  (nthr : szp { SZ.v nthr == b * (4 * cout_pg) * h_out * w_out })
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in))
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw))
  (#lbias : layout1 (4 * cout_pg))
  (#ly : layout1 (b * (4 * cout_pg) * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (convT2d_grouped_size_req b cin_pg h_in w_in cout_pg kh kw
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
       kpre #et b cin_pg h_in w_in cout_pg kh kw h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw_l sbias sy0 fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
{
  tensor_pts_to_ref gy;
  tensor_share_n gx (b * (4 * cout_pg) * h_out * w_out);
  tensor_share_n gw (b * (4 * cout_pg) * h_out * w_out);
  tensor_share_n gbias (b * (4 * cout_pg) * h_out * w_out);
  tensor_explode gy;
  forevery_iso (abs_bij #(b * (4 * cout_pg) * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * (4 * cout_pg) * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc sy0 i);
  forevery_ext
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc sy0 (abs_bij.gg i))
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias)
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * (4 * cout_pg) * h_out * w_out)) sw_l)
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_zip
    (fun (_ : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * (4 * cout_pg) * h_out * w_out)) sx)
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * (4 * cout_pg) * h_out * w_out)) sw_l) **
       (gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (b * (4 * cout_pg) * h_out * w_out) nthr;
  ()
}

(* Ghost teardown: gather N per-thread slices back into the launcher
   postcondition.  Symmetric inverse of [convt2d_grouped_naive_setup]. *)
ghost
fn convt2d_grouped_naive_teardown
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==16}) (cin_pg : szp{SZ.v cin_pg==8}) (h_in : szp{SZ.v h_in==128}) (w_in : szp{SZ.v w_in==256}) (cout_pg : szp{SZ.v cout_pg==16}) (kh : szp{SZ.v kh==3}) (kw : szp{SZ.v kw==5})
  (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==3}) (ph : sz{SZ.v ph==1}) (pw : sz{SZ.v pw==2}) (dh : szp{SZ.v dh==2}) (dw : szp{SZ.v dw==1})
  (h_out : szp{SZ.v h_out==257}) (w_out : szp{SZ.v w_out==766})
  (nthr : szp { SZ.v nthr == b * (4 * cout_pg) * h_out * w_out })
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in))
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw))
  (#lbias : layout1 (4 * cout_pg))
  (#ly : layout1 (b * (4 * cout_pg) * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#fx #fw #fb : perm)
  (#_ : squash (convT2d_grouped_size_req b cin_pg h_in w_in cout_pg kh kw
                                 sh sw ph pw dh dw h_out w_out))
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpost #et b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
             h_out w_out
             #lx #lw #lbias #ly
             gx gw gbias gy sx sw_l sbias fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw_l) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*(4*cout_pg)*h_out*w_out)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < b*(4*cout_pg)*h_out*w_out}).
               acc1 sy tid ==
               convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                              sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias tid))
{
  forevery_rw_size nthr (b * (4 * cout_pg) * h_out * w_out)
    #(kpost #et b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
            h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw_l sbias fx fw fb);
  forevery_unzip
    (fun (_ : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * (4 * cout_pg) * h_out * w_out)) sx)
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * (4 * cout_pg) * h_out * w_out)) sw_l) **
       (gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                                     sh sw ph pw dh dw
                                     h_out w_out sx sw_l sbias i));
  forevery_unzip
    (fun (_ : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * (4 * cout_pg) * h_out * w_out)) sw_l)
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                                     sh sw ph pw dh dw
                                     h_out w_out sx sw_l sbias i));
  forevery_unzip
    (fun (_ : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * (4 * cout_pg) * h_out * w_out)) sbias)
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       Cell gy (idx1 i) |-> convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                                    sh sw ph pw dh dw
                                    h_out w_out sx sw_l sbias i);
  tensor_gather_n gx (b * (4 * cout_pg) * h_out * w_out);
  tensor_gather_n gw (b * (4 * cout_pg) * h_out * w_out);
  tensor_gather_n gbias (b * (4 * cout_pg) * h_out * w_out);
  let sy : chest1 et (b * (4 * cout_pg) * h_out * w_out) =
    hide (mk1
            (fun (tid : nat{tid < b * (4 * cout_pg) * h_out * w_out}) ->
               convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                              sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias tid));
  forevery_ext
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       Cell gy (idx1 i) |-> convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                                    sh sw ph pw dh dw
                                    h_out w_out sx sw_l sbias i)
    (fun (i : natlt (b * (4 * cout_pg) * h_out * w_out)) ->
       Cell gy (abs_bij.gg i) |-> acc (reveal sy) (abs_bij.gg i));
  forevery_iso_back (abs_bij #(b * (4 * cout_pg) * h_out * w_out))
    (fun (i : Kuiper.Shape.abs ((b * (4 * cout_pg) * h_out * w_out) @| INil)) ->
       Cell gy i |-> acc (reveal sy) i);
  tensor_implode gy;
  ()
}

#pop-options

#push-options "--z3rlimit 60 --fuel 2 --ifuel 2"

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==16}) (cin_pg : szp{SZ.v cin_pg==8}) (h_in : szp{SZ.v h_in==128}) (w_in : szp{SZ.v w_in==256}) (cout_pg : szp{SZ.v cout_pg==16})
  (kh : szp{SZ.v kh==3}) (kw : szp{SZ.v kw==5})
  (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==3}) (ph : sz{SZ.v ph==1}) (pw : sz{SZ.v pw==2}) (dh : szp{SZ.v dh==2}) (dw : szp{SZ.v dw==1})
  (h_out : szp{SZ.v h_out==257}) (w_out : szp{SZ.v w_out==766})
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 (4 * cout_pg)) {| ctlayout lbias |}
  (#ly : layout1 (b * (4 * cout_pg) * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (is_global gx /\ is_global gw /\
                is_global gbias /\ is_global gy /\
                convT2d_grouped_size_req b cin_pg h_in w_in cout_pg kh kw
                                 sh sw ph pw dh dw h_out w_out))
  : kernel_desc
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw_l) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw_l) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 et (b*(4*cout_pg)*h_out*w_out)).
         (gy |-> sy) **
         pure (forall (tid : nat{tid < b*(4*cout_pg)*h_out*w_out}).
                 acc1 sy tid ==
                 convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                                sh sw ph pw dh dw
                                h_out w_out sx sw_l sbias tid)))
  = [@@inline_let] let nthr : (x : szp { SZ.v x == b * (4 * cout_pg) * h_out * w_out }) =
      b *^ 4sz *^ cout_pg *^ h_out *^ w_out in {
  nthr = nthr;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = convt2d_grouped_naive_setup b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
                                 h_out w_out nthr gx gw gbias gy;
  teardown = convt2d_grouped_naive_teardown b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
                                    h_out w_out nthr gx gw gbias gy;
  kpre  = kpre #et b cin_pg h_in w_in cout_pg kh kw h_out w_out
               #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias sy0 fx fw fb;
  kpost = kpost #et b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
                h_out w_out
                #lx #lw #lbias #ly gx gw gbias gy sx sw_l sbias fx fw fb;
  f = kf b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw h_out w_out
        gx gw gbias gy;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn convt2d_grouped_naive_gpu
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==16}) (cin_pg : szp{SZ.v cin_pg==8}) (h_in : szp{SZ.v h_in==128}) (w_in : szp{SZ.v w_in==256}) (cout_pg : szp{SZ.v cout_pg==16})
  (kh : szp{SZ.v kh==3}) (kw : szp{SZ.v kw==5})
  (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==3}) (ph : sz{SZ.v ph==1}) (pw : sz{SZ.v pw==2}) (dh : szp{SZ.v dh==2}) (dw : szp{SZ.v dw==1})
  (h_out : szp{SZ.v h_out==257}) (w_out : szp{SZ.v w_out==766})
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 (4 * cout_pg)) {| ctlayout lbias |}
  (#ly : layout1 (b * (4 * cout_pg) * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*h_out*w_out))
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
          convT2d_grouped_size_req b cin_pg h_in w_in cout_pg kh kw
                           sh sw ph pw dh dw h_out w_out)
  ensures
    (exists* (sy : chest1 et (b*(4*cout_pg)*h_out*w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*(4*cout_pg)*h_out*w_out}).
               acc1 sy tid ==
               convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                              sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias tid))
{
  launch_sync (kdesc b cin_pg h_in w_in cout_pg kh kw sh sw ph pw dh dw
                     h_out w_out gx gw gbias gy)
}
