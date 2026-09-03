module Kuiper.Kernel.Conv1D.Naive

(* Implementation of [Kuiper.Kernel.Conv1D.Naive].

   Mirrors [Kuiper.Kernel.Conv2D.Naive] with one spatial axis and an
   added [dilation] factor: the input read at tap [k] is at position
   [ol*stride + k*dilation - pad].  Setup/teardown and kpre/kpost
   sendability are all discharged.  The module is fully verified: in
   particular the [lseq_to_t3] view and its [lseq_to_t3_index] index
   lemma are proved outright (no admit). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv1D
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

let lseq_to_t3
  (#et:Type) (d0 d1 d2 : nat)
  (s : chest1 et (d0 * d1 * d2))
  : etensor3 et d0 d1 d2
  = mkT3 (fun i j k -> acc1 s ((i * d1 + j) * d2 + k))

let lseq_to_t3_index
  (#et:Type) (d0 d1 d2 : nat)
  (s : chest1 et (d0 * d1 * d2))
  (i:natlt d0) (j:natlt d1) (k:natlt d2)
  : Lemma (t3acc (lseq_to_t3 d0 d1 d2 s) i j k ==
           acc1 s ((i * d1 + j) * d2 + k))
          [SMTPat (t3acc (lseq_to_t3 d0 d1 d2 s) i j k)]
  = ()

(* Per-thread pre/post predicates. *)

unfold
let kpre
  (#et:Type) {| scalar et |}
  (b cin l_in cout : pos) (kk : pos)
  (l_out : pos)
  (#lx : layout1 (b * cin * l_in))
  (#lw : layout1 (cout * cin * kk))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * l_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*cin*l_in))
  (sw : chest1 et (cout*cin*kk))
  (sbias : chest1 et cout)
  (sy0 : chest1 et (b*cout*l_out))
  (fx fw fb : perm)
  (tid : natlt (b * cout * l_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * l_out)) sx **
    gw |-> Frac (fw /. (b * cout * l_out)) sw **
    gbias |-> Frac (fb /. (b * cout * l_out)) sbias **
    Cell gy (idx1 tid) |-> acc1 sy0 tid

unfold
let kpost
  (#et:Type) {| scalar et |}
  (b cin l_in cout : pos) (kk : pos)
  (stride : pos) (pad : nat) (dilation : pos)
  (l_out : pos)
  (#lx : layout1 (b * cin * l_in))
  (#lw : layout1 (cout * cin * kk))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * l_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (sx : chest1 et (b*cin*l_in))
  (sw : chest1 et (cout*cin*kk))
  (sbias : chest1 et cout)
  (fx fw fb : perm)
  (tid : natlt (b * cout * l_out))
  : slprop
  = gx |-> Frac (fx /. (b * cout * l_out)) sx **
    gw |-> Frac (fw /. (b * cout * l_out)) sw **
    gbias |-> Frac (fb /. (b * cout * l_out)) sbias **
    Cell gy (idx1 tid) |-> conv1d_out_at b cin l_in cout kk stride pad dilation
                                  l_out sx sw sbias tid

#push-options "--z3rlimit 60"

(* Inner-loop helper: read a tap [k] from x at [(bi, ic, l_idx)] with
   zero-padded out-of-range guards, where [l_signed = ol*stride + k*dilation]. *)
inline_for_extraction noextract
fn read_x_padded
  (#et : Type0) {| scalar et |}
  (b cin l_in : szp)
  (#lx : layout1 (b * cin * l_in)) {| ctlayout lx |}
  (gx : array1 et lx)
  (#sx : chest1 et (b*cin*l_in))
  (#fx : perm)
  (bi : szlt b)
  (ic : szlt cin)
  (l_signed : sz) (* ol*stride + k*dilation, as u32 *)
  (pad : sz)
  (#_ : squash (SZ.fits (b * cin * l_in)))
  preserves
    gpu **
    gx |-> Frac fx sx
  returns
    v : et
  ensures
    pure (
      let l_int : int = SZ.v l_signed - SZ.v pad in
      v == (if 0 <= l_int && l_int < l_in
            then acc1 sx ((bi * cin + ic) * l_in + l_int)
            else zero))
{
  if (pad <=^ l_signed) {
    let li = l_signed -^ pad;
    if (li <^ l_in) {
      let bcin = bi *^ cin;
      let bcin_ic = bcin +^ ic;
      let bcin_ic_l = bcin_ic *^ l_in;
      let flat : szlt (b * cin * l_in) = bcin_ic_l +^ li;
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

(* Read a weight tap [(oc, ic, k_i)] from the flat weight array. *)
inline_for_extraction noextract
fn read_w_tap
  (#et : Type0) {| scalar et |}
  (cout cin kk : szp)
  (#lw : layout1 (cout * cin * kk)) {| ctlayout lw |}
  (gw : array1 et lw)
  (#sw : chest1 et (cout*cin*kk))
  (#fw : perm)
  (oc : szlt cout) (ic : szlt cin) (k_i : szlt kk)
  (#_ : squash (SZ.fits (cout * cin * kk)))
  preserves
    gpu **
    gw |-> Frac fw sw
  returns
    v : et
  ensures
    pure (v == acc1 sw ((oc * cin + ic) * kk + k_i))
{
  Math.lemma_mult_lt_right cin oc cout;
  Math.lemma_mult_le_right kk (oc * cin + ic + 1) (cout * cin);
  let p1 = oc *^ cin +^ ic;
  let flat : szlt (cout * cin * kk) = p1 *^ kk +^ k_i;
  tensor_read gw (flat, ())
}

#pop-options

#push-options "--z3rlimit 60 --fuel 2 --ifuel 1"

(* Local helper: partial conv1d sum over the linearised (ic, k) index up to
   [to], with all parameters explicit.  Used as the loop-invariant predicate
   for [kf]'s accumulator.  This wrapper avoids unification difficulties
   that arise from passing the [val]-only [__conv1d_single] directly inside
   a Pulse [exists*] slprop. *)
unfold
let conv1d_partial_at
  (#et : Type) {| scalar et |}
  (b cin l_in cout : pos)
  (kk : pos) (stride : pos) (pad : nat) (dilation : pos)
  (l_out : nat)
  (sx : chest1 et (b*cin*l_in))
  (sw : chest1 et (cout*cin*kk))
  (bi : natlt b) (oc : natlt cout) (ol : natlt l_out)
  (to : nat{to <= cin * kk})
  : GTot et
  = __conv1d_single kk stride pad dilation
      (lseq_to_t3 b cin l_in sx)
      (lseq_to_t3 cout cin kk sw)
      bi oc ol to

(* Step lemma for [conv1d_partial_at]: extends the partial sum by one tap. *)
let conv1d_partial_at_step
  (#et : Type) {| scalar et |}
  (b cin l_in cout : pos)
  (kk : pos) (stride : pos) (pad : nat) (dilation : pos)
  (l_out : nat)
  (sx : chest1 et (b*cin*l_in))
  (sw : chest1 et (cout*cin*kk))
  (bi : natlt b) (oc : natlt cout) (ol : natlt l_out)
  (to : pos{to <= cin * kk})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank1_ic cin kk i in
      let k_i = unrank1_k cin kk i in
      let l_idx : int = ol * stride + k_i * dilation - pad in
      conv1d_partial_at b cin l_in cout kk stride pad dilation l_out
                        sx sw bi oc ol to ==
      add (conv1d_partial_at b cin l_in cout kk stride pad dilation l_out
                             sx sw bi oc ol (to - 1))
          (mul (read_padded1 (lseq_to_t3 b cin l_in sx) bi ic l_idx)
               (t3acc (lseq_to_t3 cout cin kk sw) oc ic k_i))))
  = __conv1d_single_lemma cin kk stride pad dilation
      (lseq_to_t3 b cin l_in sx)
      (lseq_to_t3 cout cin kk sw)
      bi oc ol to

(* Per-thread conv body: decode tid, run the inner accumulator loop,
   add bias, write to output cell.  Inherited spec-connection debt. *)
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (b cin l_in cout : szp)
  (kk : szp)
  (stride : szp) (pad : sz) (dilation : szp)
  (l_out : szp)
  (#lx : layout1 (b * cin * l_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kk)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * l_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*l_in))
  (#sw : chest1 et (cout*cin*kk))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*l_out))
  (#fx #fw #fb : perm)
  (#_ : squash (b * cout * l_out > 0))
  (#_ : squash (SZ.fits (cin * kk) /\
                SZ.fits (cout * l_out) /\
                SZ.fits (b * cin * l_in) /\
                SZ.fits (cout * cin * kk) /\
                SZ.fits (l_out * stride + kk * dilation)))
  (tid : szlt (b * cout * l_out))
  ()
  norewrite
  preserves gpu
  requires
    kpre #et b cin l_in cout kk l_out #lx #lw #lbias #ly
         gx gw gbias gy sx sw sbias sy0 fx fw fb tid
  ensures
    kpost #et b cin l_in cout kk stride pad dilation l_out #lx #lw #lbias #ly
          gx gw gbias gy sx sw sbias fx fw fb tid
{
  (* The per-thread body proves [result == conv1d_out_at ...] via a loop
     invariant tracking [acc == conv1d_partial_at ... k] and the
     step lemma [conv1d_partial_at_step] (which wraps the spec-level
     [__conv1d_single_lemma]).  Setup, teardown, and sendability are
     all discharged at the [kdesc] level (see below). *)
  let cl : sz = cout *^ l_out;
  let bi : szlt b = tid /^ cl;
  let r1 : szlt cl = tid %^ cl;
  let oc : szlt cout = r1 /^ l_out;
  let ol : szlt l_out = r1 %^ l_out;

  let n_taps : sz = cin *^ kk;
  let ol_s : sz = ol *^ stride;

  let mut acc : et = zero;
  let mut k : sz = 0sz;

  while (!k <^ n_taps)
    invariant
      exists* (vk : sz{SZ.v vk <= cin * kk}).
        k |-> vk **
        acc |-> conv1d_partial_at b cin l_in cout kk stride pad dilation
                  l_out sx sw bi oc ol vk
    invariant gx |-> Frac (fx /. (b * cout * l_out)) sx
    invariant gw |-> Frac (fw /. (b * cout * l_out)) sw
    invariant gbias |-> Frac (fb /. (b * cout * l_out)) sbias
    invariant gpu
    decreases (cin * kk - SZ.v !k)
  {
    let kk_v = !k;
    let ic : szlt cin = kk_v /^ kk;
    let k_i : szlt kk = kk_v %^ kk;

    let l_signed = ol_s +^ k_i *^ dilation;
    let xv = read_x_padded b cin l_in gx bi ic l_signed pad;
    let wv = read_w_tap cout cin kk gw oc ic k_i;
    let prod = mul xv wv;
    let acc0 = !acc;
    (* Establish the step equation: prod equals the lemma's per-tap product. *)
    assert pure (xv ==
      read_padded1 (lseq_to_t3 b cin l_in sx) bi ic
        (ol * stride + k_i * dilation - pad));
    assert pure (wv == t3acc (lseq_to_t3 cout cin kk sw) oc ic k_i);
    assert pure (SZ.v ic == unrank1_ic cin kk kk_v);
    assert pure (SZ.v k_i == unrank1_k cin kk kk_v);
    conv1d_partial_at_step b cin l_in cout kk stride pad dilation
      l_out sx sw bi oc ol (SZ.v kk_v + 1);
    acc := add acc0 prod;
    k := !k +^ 1sz;
  };

  let bias_v = tensor_read gbias (oc, ());
  let result = add bias_v !acc;
  tensor_write_cell gy (tid, ()) result
}

#pop-options

#push-options "--z3rlimit 60"

(* Ghost setup: factor the launcher's full-permission frame into N per-thread
   slices.  Mirrors [Kuiper.Kernel.Map.explode_setup_2] and
   [Kuiper.Kernel.HReduce.Max.setup_batched_max], generalised to three
   read-only fractional arrays ([gx], [gw], [gbias]) plus a full-permission
   output array ([gy]) exploded into per-cell permissions. *)
ghost
fn conv1d_naive_setup
  (#et : Type0) {| scalar et |}
  (b cin l_in cout : szp) (kk : szp)
  (stride : szp) (dilation : szp)
  (l_out : szp)
  (nthr : szp { SZ.v nthr == b * cout * l_out })
  (#lx : layout1 (b * cin * l_in))
  (#lw : layout1 (cout * cin * kk))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * l_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*l_in))
  (#sw : chest1 et (cout*cin*kk))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*l_out))
  (#fx #fw #fb : perm)
  (#_ : squash (conv1d_size_req b cin l_in cout kk stride dilation l_out))
  ()
  norewrite
  requires
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw) **
    (gbias |-> Frac fb sbias) **
    (gy |-> sy0)
  ensures
    (forall+ (tid : natlt nthr).
       kpre #et b cin l_in cout kk l_out #lx #lw #lbias #ly
            gx gw gbias gy sx sw sbias sy0 fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
{
  tensor_pts_to_ref gy;
  tensor_share_n gx (b * cout * l_out);
  tensor_share_n gw (b * cout * l_out);
  tensor_share_n gbias (b * cout * l_out);
  tensor_explode gy;
  forevery_iso (abs_bij #(b * cout * l_out))
    (fun (i : Kuiper.Shape.abs ((b * cout * l_out) @| INil)) ->
       Cell gy i |-> acc sy0 i);
  forevery_ext
    (fun (i : natlt (b * cout * l_out)) ->
       Cell gy (abs_bij.gg i) |-> acc sy0 (abs_bij.gg i))
    (fun (i : natlt (b * cout * l_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * cout * l_out)) ->
       gbias |-> Frac (fb /. (b * cout * l_out)) sbias)
    (fun (i : natlt (b * cout * l_out)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (b * cout * l_out)) ->
       gw |-> Frac (fw /. (b * cout * l_out)) sw)
    (fun (i : natlt (b * cout * l_out)) ->
       (gbias |-> Frac (fb /. (b * cout * l_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_zip
    (fun (_ : natlt (b * cout * l_out)) ->
       gx |-> Frac (fx /. (b * cout * l_out)) sx)
    (fun (i : natlt (b * cout * l_out)) ->
       (gw |-> Frac (fw /. (b * cout * l_out)) sw) **
       (gbias |-> Frac (fb /. (b * cout * l_out)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (b * cout * l_out) nthr;
  ()
}

(* Ghost teardown: gather N per-thread slices back into the launcher
   postcondition.  Symmetric inverse of [conv1d_naive_setup].  After
   gathering the cell-level permissions we exhibit the output sequence
   via [Seq.init], whose [lemma_init_index] discharges the per-cell
   functional equation in the launcher's pure postcondition. *)
ghost
fn conv1d_naive_teardown
  (#et : Type0) {| scalar et |}
  (b cin l_in cout : szp) (kk : szp)
  (stride : szp) (pad : sz) (dilation : szp)
  (l_out : szp)
  (nthr : szp { SZ.v nthr == b * cout * l_out })
  (#lx : layout1 (b * cin * l_in))
  (#lw : layout1 (cout * cin * kk))
  (#lbias : layout1 cout)
  (#ly : layout1 (b * cout * l_out))
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*l_in))
  (#sw : chest1 et (cout*cin*kk))
  (#sbias : chest1 et cout)
  (#fx #fw #fb : perm)
  (#_ : squash (conv1d_size_req b cin l_in cout kk stride dilation l_out))
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpost #et b cin l_in cout kk stride pad dilation l_out
             #lx #lw #lbias #ly
             gx gw gbias gy sx sw sbias fx fw fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gx |-> Frac fx sx) **
    (gw |-> Frac fw sw) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b*cout*l_out)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*l_out}).
               acc1 sy tid ==
               conv1d_out_at b cin l_in cout kk stride pad dilation
                             l_out sx sw sbias tid))
{
  forevery_rw_size nthr (b * cout * l_out)
    #(kpost #et b cin l_in cout kk stride pad dilation l_out
            #lx #lw #lbias #ly
            gx gw gbias gy sx sw sbias fx fw fb);
  forevery_unzip
    (fun (_ : natlt (b * cout * l_out)) -> gx |-> Frac (fx /. (b * cout * l_out)) sx)
    (fun (i : natlt (b * cout * l_out)) ->
       (gw |-> Frac (fw /. (b * cout * l_out)) sw) **
       (gbias |-> Frac (fb /. (b * cout * l_out)) sbias) **
       (Cell gy (idx1 i) |-> conv1d_out_at b cin l_in cout kk stride pad dilation
                                    l_out sx sw sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * l_out)) -> gw |-> Frac (fw /. (b * cout * l_out)) sw)
    (fun (i : natlt (b * cout * l_out)) ->
       (gbias |-> Frac (fb /. (b * cout * l_out)) sbias) **
       (Cell gy (idx1 i) |-> conv1d_out_at b cin l_in cout kk stride pad dilation
                                    l_out sx sw sbias i));
  forevery_unzip
    (fun (_ : natlt (b * cout * l_out)) -> gbias |-> Frac (fb /. (b * cout * l_out)) sbias)
    (fun (i : natlt (b * cout * l_out)) ->
       Cell gy (idx1 i) |-> conv1d_out_at b cin l_in cout kk stride pad dilation
                                   l_out sx sw sbias i);
  tensor_gather_n gx (b * cout * l_out);
  tensor_gather_n gw (b * cout * l_out);
  tensor_gather_n gbias (b * cout * l_out);
  let sy : chest1 et (b * cout * l_out) =
    hide (mk1
            (fun (tid : nat{tid < b * cout * l_out}) ->
               conv1d_out_at b cin l_in cout kk stride pad dilation
                             l_out sx sw sbias tid));
  forevery_ext
    (fun (i : natlt (b * cout * l_out)) ->
       Cell gy (idx1 i) |-> conv1d_out_at b cin l_in cout kk stride pad dilation
                                   l_out sx sw sbias i)
    (fun (i : natlt (b * cout * l_out)) ->
       Cell gy (abs_bij.gg i) |-> acc (reveal sy) (abs_bij.gg i));
  forevery_iso_back (abs_bij #(b * cout * l_out))
    (fun (i : Kuiper.Shape.abs ((b * cout * l_out) @| INil)) ->
       Cell gy i |-> acc (reveal sy) i);
  tensor_implode gy;
  ()
}

#pop-options

#push-options "--z3rlimit 60 --fuel 2 --ifuel 2"

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (b cin l_in cout : szp)
  (kk : szp)
  (stride : szp) (pad : sz) (dilation : szp)
  (l_out : szp)
  (#lx : layout1 (b * cin * l_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kk)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * l_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*l_in))
  (#sw : chest1 et (cout*cin*kk))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*l_out))
  (#fx #fw #fb : perm)
  (#_ : squash (is_global gx /\ is_global gw /\
                is_global gbias /\ is_global gy /\
                conv1d_size_req b cin l_in cout kk stride dilation l_out))
  : kernel_desc
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gx |-> Frac fx sx) **
       (gw |-> Frac fw sw) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 et (b*cout*l_out)).
         (gy |-> sy) **
         pure (forall (tid : nat{tid < b*cout*l_out}).
                 acc1 sy tid ==
                 conv1d_out_at b cin l_in cout kk stride pad dilation
                               l_out sx sw sbias tid)))
  = [@@inline_let] let nthr : (x : szp { SZ.v x == b * cout * l_out }) =
      b *^ cout *^ l_out in {
  nthr = nthr;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = conv1d_naive_setup b cin l_in cout kk stride dilation l_out nthr
                                gx gw gbias gy;
  teardown = conv1d_naive_teardown b cin l_in cout kk stride pad dilation l_out nthr
                                   gx gw gbias gy;
  kpre  = kpre #et b cin l_in cout kk l_out #lx #lw #lbias #ly gx gw gbias gy sx sw sbias sy0 fx fw fb;
  kpost = kpost #et b cin l_in cout kk stride pad dilation l_out #lx #lw #lbias #ly gx gw gbias gy sx sw sbias fx fw fb;
  f = kf b cin l_in cout kk stride pad dilation l_out gx gw gbias gy;
  (* Inherited tree-wide debt (sendability of compound per-thread slprops). *)
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn conv1d_naive_gpu
  (#et : Type0) {| scalar et |}
  (b cin l_in cout : szp)
  (kk : szp)
  (stride : szp) (pad : sz) (dilation : szp)
  (l_out : szp)
  (#lx : layout1 (b * cin * l_in)) {| ctlayout lx |}
  (#lw : layout1 (cout * cin * kk)) {| ctlayout lw |}
  (#lbias : layout1 cout) {| ctlayout lbias |}
  (#ly : layout1 (b * cout * l_out)) {| ctlayout ly |}
  (gx : array1 et lx)
  (gw : array1 et lw)
  (gbias : array1 et lbias)
  (gy : array1 et ly)
  (#sx : chest1 et (b*cin*l_in))
  (#sw : chest1 et (cout*cin*kk))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b*cout*l_out))
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
          conv1d_size_req b cin l_in cout kk stride dilation l_out)
  ensures
    (exists* (sy : chest1 et (b*cout*l_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b*cout*l_out}).
               acc1 sy tid ==
               conv1d_out_at b cin l_in cout kk stride pad dilation
                             l_out sx sw sbias tid))
{
  launch_sync (kdesc b cin l_in cout kk stride pad dilation l_out gx gw gbias gy)
}
