module Kuiper.Kernel.Conv2D.Naive

(* Implementation of [Kuiper.Kernel.Conv2D.Naive].

   See the [.fsti] for the contract and the spec correspondence.  The
   kernel computes one output pixel per thread; the inner accumulation
   over the [(ic, kh_i, kw_i)] taps is a single while-loop matched up
   to [Kuiper.Spec.Conv2D.__conv2d_single].

   Setup, teardown, and kpre/kpost sendability are all discharged at
   the [kdesc] level. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv2D
open FStar.FunctionalExtensionality { (^->>) }
open Kuiper.Bijection { ( =~ ) }
module Seq = FStar.Seq
module SZ = Kuiper.SizeT

let flatten4_index_bound
  (b cin h w : pos)
  (bi ic hi wi : nat)
  : Lemma
      (requires bi < b /\ ic < cin /\ hi < h /\ wi < w)
      (ensures (((bi * cin + ic) * h + hi) * w + wi) < b * cin * h * w)
  = ()

let decreases_after_increment (bound k : nat)
  : Lemma (requires k < bound) (ensures (bound - (k + 1) < bound - k))
  = ()
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

(* Total wrapper around [__conv2d_single] that accepts arbitrary nats and
   clamps internally — avoids refinement-typing obligations in invariants. *)
let conv2d_acc
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat) (#cout : nat)
  (kh : nat) (kw : nat)
  (stride : nat) (pad : nat)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : nat)
  : GTot et
  = if kh = 0 || kw = 0 || stride = 0 || cin = 0 then zero
    else
      let kh' : pos = kh in
      let kw' : pos = kw in
      let stride' : pos = stride in
      let to' : nat = if to > cin * kh' * kw' then cin * kh' * kw' else to in
      let to'' : (n:nat{n <= (if cin = 0 then 0 else cin * kh' * kw')}) = to' in
      __conv2d_single kh' kw' stride' pad x weight b oc oh ow to''

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
  (sw : chest1 et (cout*cin*kh*kw))
  (sbias : chest1 et cout)
  (sy0 : chest1 et (b*cout*h_out*w_out))
  (fx fw fb : perm)
  (tid : natlt (b * cout * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw **
    gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> acc1 sy0 tid

unfold
let kpost
  (#et:Type) {| scalar et |}
  (b cin h_in w_in cout : pos) (kh kw : pos)
  (stride : pos) (pad : nat)
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
  (sw : chest1 et (cout*cin*kh*kw))
  (sbias : chest1 et cout)
  (fx fw fb : perm)
  (tid : natlt (b * cout * h_out * w_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx **
    gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw **
    gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias **
    Cell gy (idx1 tid) |-> conv2d_out_at b cin h_in w_in cout kh kw stride pad
                                  h_out w_out sx sw sbias tid

#push-options "--z3rlimit 120"

(* Inner-loop helper: read a tap [k] from x at [(bi, ic, h_idx, w_idx)],
   with zero-padded out-of-range guards, where [(ic, kh_i, kw_i)] are the
   linearised conv taps and [h_idx = oh*stride + kh_i - pad],
   [w_idx = ow*stride + kw_i - pad]. *)
inline_for_extraction noextract
fn read_x_padded
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in : szp)
  (#lx : layout1 (b * cin * h_in * w_in)) {| ctlayout lx |}
  (gx : array1 et lx)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#fx : perm)
  (bi : szlt b)
  (ic : szlt cin)
  (h_signed : sz) (* oh*stride + kh_i, in u32 *)
  (w_signed : sz) (* ow*stride + kw_i, in u32 *)
  (pad : sz)
  (#_ : squash (SZ.fits (b * cin * h_in * w_in)))
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
            then acc1 sx (((bi * cin + ic) * h_in + h_int) * w_in + w_int)
            else zero))
{
  if (pad <=^ h_signed && pad <=^ w_signed) {
    let hi = h_signed -^ pad;
    let wi = w_signed -^ pad;
    if (hi <^ h_in && wi <^ w_in) {
      flatten4_index_bound b cin h_in w_in bi ic
        hi wi;
      // Compute flat index: ((bi*cin + ic)*h_in + hi)*w_in + wi
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
  (#sw : chest1 et (cout*cin*kh*kw))
  (#fw : perm)
  (oc : szlt cout) (ic : szlt cin)
  (kh_i : szlt kh) (kw_i : szlt kw)
  (#_ : squash (SZ.fits (cout * cin * kh * kw)))
  preserves
    gpu **
    gw |-> Frac fw sw
  returns
    v : et
  ensures
    pure (v == acc1 sw (((oc * cin + ic) * kh + kh_i) * kw + kw_i))
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

(* Local helper: partial conv2d sum over the linearised (ic, kh_i, kw_i)
   index up to [to], with all parameters explicit.  Used as the loop-
   invariant predicate for [kf]'s accumulator.  This wrapper avoids
   unification difficulties that arise from passing the [val]-only
   [__conv2d_single] directly inside a Pulse [exists*] slprop. *)
unfold
let conv2d_partial_at
  (#et : Type) {| scalar et |}
  (b cin h_in w_in cout : pos)
  (kh kw : pos) (stride : pos) (pad : nat)
  (h_out w_out : nat)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw : chest1 et (cout*cin*kh*kw))
  (bi : natlt b) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= cin * kh * kw})
  : GTot et
  = __conv2d_single kh kw stride pad
      (lseq_to_t4 b cin h_in w_in sx)
      (lseq_to_t4 cout cin kh kw sw)
      bi oc oh ow to

(* Step lemma for [conv2d_partial_at]: extends the partial sum by one tap. *)
let conv2d_partial_at_step
  (#et : Type) {| scalar et |}
  (b cin h_in w_in cout : pos)
  (kh kw : pos) (stride : pos) (pad : nat)
  (h_out w_out : nat)
  (sx : chest1 et (b*cin*h_in*w_in))
  (sw : chest1 et (cout*cin*kh*kw))
  (bi : natlt b) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank_ic cin kh kw i in
      let kh_i = unrank_kh cin kh kw i in
      let kw_i = unrank_kw cin kh kw i in
      let h_idx : int = oh * stride + kh_i - pad in
      let w_idx : int = ow * stride + kw_i - pad in
      conv2d_partial_at b cin h_in w_in cout kh kw stride pad h_out w_out
                        sx sw bi oc oh ow to ==
      add (conv2d_partial_at b cin h_in w_in cout kh kw stride pad h_out w_out
                             sx sw bi oc oh ow (to - 1))
          (mul (read_padded (lseq_to_t4 b cin h_in w_in sx) bi ic h_idx w_idx)
               (tacc (lseq_to_t4 cout cin kh kw sw) oc ic kh_i kw_i))))
  = __conv2d_single_lemma cin kh kw stride pad
      (lseq_to_t4 b cin h_in w_in sx)
      (lseq_to_t4 cout cin kh kw sw)
      bi oc oh ow to

(* Per-thread conv body: decode tid, run the inner accumulator loop,
   add bias, write to output cell.  Carries the [__conv2d_single]
   spec invariant through the loop. *)
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (stride : szp) (pad : sz)
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
  (#sw : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (b * cout * h_out * w_out > 0))
  (#_ : squash (SZ.fits (cin * kh * kw) /\
                SZ.fits (h_out * w_out) /\
                SZ.fits (cout * h_out * w_out) /\
                SZ.fits (b * cin * h_in * w_in) /\
                SZ.fits (cout * cin * kh * kw) /\
                SZ.fits (h_out * stride + kh) /\
                SZ.fits (w_out * stride + kw)))
  (tid : szlt (b * cout * h_out * w_out))
  ()
  norewrite
  preserves gpu
  requires
    kpre #et b cin h_in w_in cout kh kw h_out w_out #lx #lw #lbias #ly
         gx gw gbias gy sx sw sbias sy0 fx fw fb tid
  ensures
    kpost #et b cin h_in w_in cout kh kw stride pad h_out w_out #lx #lw #lbias #ly
          gx gw gbias gy sx sw sbias fx fw fb tid
{
  (* The per-thread body proves [result == conv2d_out_at ...] via a loop
     invariant tracking [acc == conv2d_partial_at ... k] and the
     step lemma [conv2d_partial_at_step] (which wraps the spec-level
     [__conv2d_single_lemma]).  Setup, teardown, and sendability are
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

  let oh_s : sz = oh *^ stride;
  let ow_s : sz = ow *^ stride;

  let mut acc : et = zero;
  let mut k : sz = 0sz;

  while (!k <^ n_taps)
    invariant
      exists* (vk : sz{SZ.v vk <= cin * kh * kw}).
        k |-> vk **
        acc |-> conv2d_partial_at b cin h_in w_in cout kh kw stride pad
                  h_out w_out sx sw bi oc oh ow vk
    invariant gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx
    invariant gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw
    invariant gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias
    invariant gpu
    decreases (cin * kh * kw - SZ.v !k)
  {
    let kk_v = !k;
    let ic : szlt cin = kk_v /^ kh_kw;
    let r : szlt kh_kw = kk_v %^ kh_kw;
    let kh_i : szlt kh = r /^ kw;
    let kw_i : szlt kw = r %^ kw;

    let h_signed = oh_s +^ kh_i;
    let w_signed = ow_s +^ kw_i;
    let xv = read_x_padded b cin h_in w_in gx bi ic h_signed w_signed pad;
    let wv = read_w_tap cout cin kh kw gw oc ic kh_i kw_i;
    let prod = mul xv wv;
    let acc0 = !acc;
    (* Establish the step equation: prod equals the lemma's per-tap product. *)
    assert pure (xv ==
      read_padded (lseq_to_t4 b cin h_in w_in sx) bi ic
        (oh * stride + kh_i - pad) (ow * stride + kw_i - pad));
    assert pure (wv == tacc (lseq_to_t4 cout cin kh kw sw) oc ic kh_i kw_i);
    assert pure (SZ.v ic == unrank_ic cin kh kw kk_v);
    assert pure (SZ.v kh_i == unrank_kh cin kh kw kk_v);
    assert pure (SZ.v kw_i == unrank_kw cin kh kw kk_v);
    conv2d_partial_at_step b cin h_in w_in cout kh kw stride pad
      h_out w_out sx sw bi oc oh ow (SZ.v kk_v + 1);
    acc := add acc0 prod;
    assert pure (SZ.v n_taps == SZ.v cin * SZ.v kh * SZ.v kw);
    assert pure (SZ.v kk_v < SZ.v cin * SZ.v kh * SZ.v kw);
    decreases_after_increment (SZ.v cin * SZ.v kh * SZ.v kw) kk_v;
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
   slices.  Mirrors [Kuiper.Kernel.Conv1D.Naive.conv1d_naive_setup] adapted
   to Conv2D's 4-D output indexing: three read-only fractional arrays
   ([gx], [gw], [gbias]) plus a full-permission output array ([gy]) exploded
   into per-cell permissions over [b * cout * h_out * w_out] threads. *)
ghost
fn conv2d_naive_setup
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (stride : szp)
  (h_out w_out : szp)
  (nthr : szp { SZ.v nthr == b * cout * h_out * w_out })
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cout * cin * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out))
  ()
  norewrite
  requires
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw) **
    (gbias |-> Frac fb sbias) **
    (gy |-> sy0)
  ensures
    (forall+ (tid : natlt nthr).
       kpre #et b cin h_in w_in cout kh kw h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw sbias sy0 fx fw fb tid) **
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
       gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_zip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw) **
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (b * cout * h_out * w_out) nthr;
  ()
}

(* Ghost teardown: gather N per-thread slices back into the launcher
   postcondition.  Symmetric inverse of [conv2d_naive_setup].  After
   gathering the cell-level permissions we exhibit the output sequence
   via [Seq.init_ghost], whose [lemma_init_ghost_index] discharges the
   per-cell functional equation in the launcher's pure postcondition. *)
ghost
fn conv2d_naive_teardown
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (stride : szp) (pad : sz)
  (h_out w_out : szp)
  (nthr : szp { SZ.v nthr == b * cout * h_out * w_out })
  (#lx : layout1 (b * cin * h_in * w_in))
  (#lw : layout1 (cout * cin * kh * kw))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * h_out * w_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*h_in*w_in))
  (#sw : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#fx #fw #fb : perm)
  (#_ : squash (conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out))
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpost #et b cin h_in w_in cout kh kw stride pad h_out w_out
             #lx #lw #lbias #ly
             gx gw gbias gy sx sw sbias fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*h_out*w_out)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
               acc1 sy tid ==
               conv2d_out_at b cin h_in w_in cout kh kw stride pad
                             h_out w_out sx sw sbias tid))
{
  forevery_rw_size nthr (b * cout * h_out * w_out)
    #(kpost #et b cin h_in w_in cout kh kw stride pad h_out w_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw sbias fx fw fb);
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gx |-> Frac (fx /. (b * cout * h_out * w_out)) sx)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw) **
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> conv2d_out_at b cin h_in w_in cout kh kw stride pad
                                    h_out w_out sx sw sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gw |-> Frac (fw /. (b * cout * h_out * w_out)) sw)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       (gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias) **
       (Cell gy (idx1 i) |-> conv2d_out_at b cin h_in w_in cout kh kw stride pad
                                    h_out w_out sx sw sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * h_out * w_out)) ->
       gbias |-> Frac (fb /. (b * cout * h_out * w_out)) sbias)
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (idx1 i) |-> conv2d_out_at b cin h_in w_in cout kh kw stride pad
                                   h_out w_out sx sw sbias i);
  tensor_gather_n gx (b * cout * h_out * w_out);
  tensor_gather_n gw (b * cout * h_out * w_out);
  tensor_gather_n gbias (b * cout * h_out * w_out);
  let sy : chest1 et (b * cout * h_out * w_out) =
    hide (mk1
            (fun (tid : nat{tid < b * cout * h_out * w_out}) ->
               conv2d_out_at b cin h_in w_in cout kh kw stride pad
                             h_out w_out sx sw sbias tid));
  forevery_ext
    (fun (i : natlt (b * cout * h_out * w_out)) ->
       Cell gy (idx1 i) |-> conv2d_out_at b cin h_in w_in cout kh kw stride pad
                                   h_out w_out sx sw sbias i)
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
  (stride : szp) (pad : sz)
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
  (#sw : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  (#_ : squash (is_global gx /\ is_global gw /\
                is_global gbias /\ is_global gy /\
                conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out))
  : kernel_desc
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 et (b*cout*h_out*w_out)).
         (gy |-> sy) **
         pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
                 acc1 sy tid ==
                 conv2d_out_at b cin h_in w_in cout kh kw stride pad
                               h_out w_out sx sw sbias tid)))
  = [@@inline_let] let nthr : (x : szp { SZ.v x == b * cout * h_out * w_out }) =
      b *^ cout *^ h_out *^ w_out in {
  nthr = nthr;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = conv2d_naive_setup b cin h_in w_in cout kh kw stride h_out w_out nthr
                                gx gw gbias gy;
  teardown = conv2d_naive_teardown b cin h_in w_in cout kh kw stride pad
                                   h_out w_out nthr gx gw gbias gy;
  kpre  = kpre #et b cin h_in w_in cout kh kw h_out w_out #lx #lw #lbias #ly gx gw gbias gy sx sw sbias sy0 fx fw fb;
  kpost = kpost #et b cin h_in w_in cout kh kw stride pad h_out w_out #lx #lw #lbias #ly gx gw gbias gy sx sw sbias fx fw fb;
  f = kf b cin h_in w_in cout kh kw stride pad h_out w_out gx gw gbias gy;
  (* Inherited tree-wide debt (sendability of compound per-thread slprops). *)
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn conv2d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout : szp)
  (kh kw : szp)
  (stride : szp) (pad : sz)
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
  (#sw : chest1 et (cout*cin*kh*kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*h_out*w_out))
  (#fx #fw #fb : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\
          is_global gbias /\ is_global gy /\
          conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out)
  ensures
    (exists* (sy : chest1 et (b*cout*h_out*w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*h_out*w_out}).
               acc1 sy tid ==
               conv2d_out_at b cin h_in w_in cout kh kw stride pad
                             h_out w_out sx sw sbias tid))
{
  launch_sync (kdesc b cin h_in w_in cout kh kw stride pad h_out w_out gx gw gbias gy)
}
