module Kuiper.KB.TripletMarginLoss

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.TripletMarginLoss
open Kuiper.Seq.Common { (@!) }
module SZ = Kuiper.SizeT
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module KBMap = Kuiper.KB.Compat.Map
module Vec = Pulse.Lib.Vec
module KS = Kuiper.Seq.Common

(* Verified reciprocal 1/B as f32. This is inlined into the public entry point,
   so callers cannot supply a scale inconsistent with [b]. *)
inline_for_extraction noextract
let triplet_recip_f32 (b : szp) : f32 =
  div one (of_int (FStar.Int.Cast.uint64_to_int64
                     (FStar.SizeT.sizet_to_uint64 b)))

(* [lseq_map2_index] is proved locally for the KuiperBench compatibility helper.
   [lseq_map2 f sa sb] is [Seq.init_ghost], so indexing it at
   [i] yields [f (sa @! i) (sb @! i)] definitionally. *)
let lseq_map2_index
  (#a #b #c : Type0)
  (#len : nat)
  (f : a -> b -> c)
  (sa : Seq.lseq a len) (sb : Seq.lseq b len)
  (i : nat{i < len})
  : Lemma (Seq.index (KBMap.lseq_map2 f sa sb) i == f (sa @! i) (sb @! i))
  = ()

(* Manually-inlined squared-difference and triplet step.  We avoid
   the polymorphic [scalar] typeclass at the value level for the same
   reason as in [Kuiper.KB.HingeLoss]: Karamel sometimes fails to
   project [zero] / [sub] / [mul] from the dictionary at extraction
   time. *)
inline_for_extraction noextract
let sq_diff_step_f32 (eps x y : f32) : f32 =
  let d = add (sub x y) eps in mul d d

inline_for_extraction noextract
let triplet_step_f32 (margin d_ap d_an : f32) : f32 =
  fmax (zero #f32) (add (sub d_ap d_an) margin)

let triplet_step_f32_eq (margin d_ap d_an : f32)
  : Lemma (triplet_step_f32 margin d_ap d_an ==
           triplet_step margin d_ap d_an)
  = ()

let seq_map_id_eq (#a:Type) (s : Seq.seq a)
  : Lemma (Seq.equal (KS.seq_map id s) s)
  = ()

(* ─────────────────────────────────────────────────────────────────────
   chest/seq bridge lemmas + tensor-level device memcpy helpers.  These
   wrap the core [larray] [gpu_memcpy_*] primitives with the
   [tensor_concr]/[tensor_abs] bridge (cf. Kuiper.Kernel.LogSoftmax),
   exposing postconditions in [chest1_to_seq] / [acc1] terms.  (Same
   helper block as in Kuiper.KB.CrossEntropyLoss; duplicated here because
   a shared subdir module would not be on the (non-recursive) include
   path.)
   ───────────────────────────────────────────────────────────────────── *)
let lem_to_seq (#et:Type) (n:nat) (c : chest1 et n)
  : Lemma (to_seq (l1_forward n) c == chest1_to_seq c)
  = assert (Seq.equal (to_seq (l1_forward n) c) (chest1_to_seq c))

let lem_index_chest1 (#et:Type) (#n:nat) (c : chest1 et n) (i:natlt n)
  : Lemma (Seq.index (chest1_to_seq c) i == acc1 c i)
  = ()

(* [chest1_to_seq] is a left inverse of [seq_to_chest1]. *)
let chest1_seq_roundtrip (#et:Type) (#n:nat) (s : lseq et n)
  : Lemma (Seq.equal (chest1_to_seq (seq_to_chest1 s)) s)
  = ()

let lem_to_real_chest_to_seq (#et:Type0) {| scalar et, real_like et |} (#n:nat)
  (c : chest1 et n)
  : Lemma (chest1_to_seq (to_real_chest c) == to_real_seq (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (to_real_chest c)) (to_real_seq (chest1_to_seq c)))

(* device-to-device blit (chest level) *)
inline_for_extraction noextract
fn t_memcpy_d2d'
  (#a:Type u#0) {| sized a |}
  (#dst_sz : erased nat)
  (dst : array1 a (l1_forward dst_sz))
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src : array1 a (l1_forward src_sz))
  (src_off : SZ.t)
  (cnt : SZ.t { SZ.v dst_off + SZ.v cnt <= dst_sz /\ SZ.v src_off + SZ.v cnt <= src_sz })
  (#f : perm)
  (#v : chest1 a src_sz)
  (#gv : chest1 a dst_sz)
  preserves cpu ** on gpu_loc (src |-> Frac f v)
  requires on gpu_loc (dst |-> gv)
  ensures exists* (s' : chest1 a dst_sz).
      on gpu_loc (dst |-> s') **
      pure (chest1_to_seq s' ==
            KS.seq_blit (chest1_to_seq gv) dst_off (chest1_to_seq v) src_off cnt)
{
  map_loc gpu_loc #(dst |-> gv) #(core dst |-> to_seq (l1_forward dst_sz) gv)
    fn _ { tensor_concr dst; };
  map_loc gpu_loc #(src |-> Frac f v) #(core src |-> Frac f (to_seq (l1_forward src_sz) v))
    fn _ { tensor_concr src; };
  Kuiper.KB.Compat.Array.gpu_memcpy_device_to_device' (core dst) dst_off (core src) src_off cnt;
  with s'seq. assert (on gpu_loc (core dst |-> s'seq));
  map_loc gpu_loc #(core src |-> Frac f (to_seq (l1_forward src_sz) v)) #(src |-> Frac f v)
    fn _ {
      tensor_abs (l1_forward src_sz) (core src);
      rewrite (from_array (l1_forward src_sz) (core src) |-> Frac f v)
           as (src |-> Frac f v);
    };
  map_loc gpu_loc #(core dst |-> s'seq) #(dst |-> from_seq (l1_forward dst_sz) s'seq)
    fn _ {
      tensor_abs' (l1_forward dst_sz) (core dst);
      rewrite (from_array (l1_forward dst_sz) (core dst) |-> from_seq (l1_forward dst_sz) s'seq)
           as (dst |-> from_seq (l1_forward dst_sz) s'seq);
    };
  lem_to_seq dst_sz gv;
  lem_to_seq src_sz v;
  lem_to_seq dst_sz (from_seq (l1_forward dst_sz) s'seq);
  ()
}

(* host vec -> device (full) *)
inline_for_extraction noextract
fn t_memcpy_h2d
  (#a:Type u#0) {| sized a |}
  (#sz : erased nat)
  (dst : array1 a (l1_forward sz))
  (src : Vec.vec a)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (Seq.seq a))
  (#gv : chest1 a sz)
  preserves cpu ** (src |-> Frac f v)
  requires on gpu_loc (dst |-> gv) **
    pure (SZ.v cnt == sz /\ (Vec.length src == sz \/ Seq.length v == reveal sz))
  ensures exists* (s' : chest1 a sz).
    on gpu_loc (dst |-> s') **
    pure (Seq.length v == reveal sz /\ chest1_to_seq s' == reveal v)
{
  map_loc gpu_loc #(dst |-> gv) #(core dst |-> to_seq (l1_forward sz) gv)
    fn _ { tensor_concr dst; };
  gpu_memcpy_host_to_device (core dst) src cnt;
  let vl : erased (Seq.lseq a sz) = hide (reveal v <: Seq.lseq a sz);
  map_loc gpu_loc #(core dst |-> reveal vl) #(dst |-> from_seq (l1_forward sz) vl)
    fn _ {
      tensor_abs' (l1_forward sz) (core dst);
      rewrite (from_array (l1_forward sz) (core dst) |-> from_seq (l1_forward sz) vl)
           as (dst |-> from_seq (l1_forward sz) vl);
    };
  lem_to_seq sz (from_seq (l1_forward sz) vl);
  ()
}

(* ── Approximation lemmas linking the f32 squared-difference step to
      the real-arithmetic [sqdiff_step_r]. ─────────────────────────── *)

(* Pointwise: the f32 epsilon-shifted squared difference approximates
   the corresponding real expression. [sub_approx] and [a_add] give
   [(x - y + eps) %~ (rx -. ry +. to_real eps)], then [a_mul] squares it. *)
let sqdiff_approx (eps x y : f32) (rx ry : real)
  : Lemma (requires v_approximates x rx /\ v_approximates y ry)
          (ensures  v_approximates (sq_diff_step_f32 eps x y)
                                  (sqdiff_step_r eps rx ry))
  = to_real_ok eps;
    sub_approx x y rx ry;
    a_add (sub x y) eps (rx -. ry) (to_real eps);
    a_mul (add (sub x y) eps) (add (sub x y) eps)
          ((rx -. ry) +. to_real eps) ((rx -. ry) +. to_real eps)

(* Sequence-level lift: the elementwise f32 squared-difference of two
   rows approximates the elementwise real squared-difference of their
   [to_real] images. *)
let map2_sqdiff_approx (eps : f32) (d:nat) (sca scb : Seq.lseq f32 d)
  : Lemma
      (seq_approximates
        (KBMap.lseq_map2 (sq_diff_step_f32 eps) sca scb)
        (Seq.init d (fun j ->
          sqdiff_step_r eps (to_real (sca @! j)) (to_real (scb @! j)))))
  = let lhs : Seq.lseq f32 d = KBMap.lseq_map2 (sq_diff_step_f32 eps) sca scb in
    let rhs : Seq.lseq real d =
      Seq.init d (fun j ->
        sqdiff_step_r eps (to_real (sca @! j)) (to_real (scb @! j))) in
    introduce forall (i:nat{i < Seq.length lhs}). (lhs @! i) %~ (rhs @! i)
    with begin
      to_real_ok (sca @! i);
      to_real_ok (scb @! i);
      lseq_map2_index (sq_diff_step_f32 eps) sca scb i;
      sqdiff_approx eps (sca @! i) (scb @! i)
        (to_real (sca @! i)) (to_real (scb @! i))
    end;
    assert (Seq.length lhs == Seq.length rhs)

(* Chest-level analogue of [map2_sqdiff_approx], phrased directly over
   the [acc1]-values of two GPU-ownership chests.  [sqdiff_row_real sca
   scb] is the real-arithmetic integrand of the squared distance; the
   f32 elementwise map [chest1_map2 sq_diff_step_f32] approximates its
   [seq_to_chest1] lift.  (Mirrors [Kuiper.KB.MSELoss.mse_map_approx].) *)
let sqdiff_row_real (eps : f32) (dd:nat) (sca scb : chest1 f32 dd)
  : GTot (lseq real dd) =
  Seq.init_ghost dd (fun j ->
    sqdiff_step_r eps (to_real (acc1 sca j)) (to_real (acc1 scb j)))

(* Pointwise-to-chest approximation intro (mirrors
   [Kuiper.Kernel.OnlineSoftmax.chest1_approx_intro]): the explicit
   [let (b0, ()) = i] destructuring discharges the [natlt]-to-[abs]
   bridge that [Classical.forall_intro] alone leaves open. *)
let chest1_approx_intro
  (#et : Type0) {| scalar et, real_like et |} (#n : nat)
  (c1 : chest1 et n) (c2 : chest1 real n)
  : Lemma (requires forall (bid:natlt n). acc1 c1 bid %~ acc1 c2 bid)
          (ensures c1 %~ c2)
  = introduce forall (i:abs (n @| INil)). acc c1 i %~ acc c2 i
    with (let (b0, ()) = i in ())

#push-options ""
let sqdiff_map_approx (eps : f32) (dd:nat) (sca scb : chest1 f32 dd)
  : Lemma (Map.chest1_map2 (sq_diff_step_f32 eps) sca scb
           %~ seq_to_chest1 (sqdiff_row_real eps dd sca scb))
  = let c1 = Map.chest1_map2 (sq_diff_step_f32 eps) sca scb in
    let c2 = seq_to_chest1 (sqdiff_row_real eps dd sca scb) in
    let aux (i:natlt dd)
      : Lemma (acc1 c1 i %~ acc1 c2 i)
      = to_real_ok (acc1 sca i);
        to_real_ok (acc1 scb i);
        sqdiff_approx eps (acc1 sca i) (acc1 scb i)
          (to_real (acc1 sca i)) (to_real (acc1 scb i))
    in
    Classical.forall_intro aux;
    chest1_approx_intro c1 c2
#pop-options

(* The real integrand of a copied row, re-expressed over the (equal)
   underlying sequences [rx]/[ry]; lets [real_sq_dist_unfold] line up. *)
let sqdiff_row_real_eq (eps : f32) (dd:nat) (sca scb : chest1 f32 dd)
  (rx ry : Seq.lseq f32 dd)
  : Lemma (requires chest1_to_seq sca == rx /\ chest1_to_seq scb == ry)
          (ensures sqdiff_row_real eps dd sca scb ==
                   Seq.init dd (fun j ->
                     sqdiff_step_r eps (to_real (rx @! j)) (to_real (ry @! j))))
  = assert (Seq.equal (sqdiff_row_real eps dd sca scb)
              (Seq.init dd (fun j ->
                sqdiff_step_r eps (to_real (rx @! j)) (to_real (ry @! j)))))

(* [real_sq_dist] unfolded to its [rsum]-of-[Seq.init] form, so the
   SMT solver does not have to unfold the [let] on its own in the
   (large) kernel proof context. *)
let real_sq_dist_unfold (eps : f32) (d:nat) (ra rb : Seq.lseq f32 d)
  : Lemma
    (real_sq_dist eps d ra rb ==
     rsum (Seq.init d (fun j ->
       sqdiff_step_r eps (to_real (ra @! j)) (to_real (rb @! j)))))
  = ()

(* ── Carried loop-invariant predicate. ─────────────────────────────── *)

let tup4 (b:nat) =
  (Seq.lseq f32 b & Seq.lseq f32 b & Seq.lseq f32 b & Seq.lseq f32 b)

(* Total accessor for the (host) result vector: keeps [carried_pred]
   total even when its length is left abstract. *)
let sidx (s : Seq.seq f32) (r : nat) : f32 =
  if r < Seq.length s then Seq.index s r else zero

(* For every already-processed row [r < bound]:
     - [d_ap[r] == sqrt sumsq_ap[r]] / [d_an[r] == sqrt sumsq_an[r]];
     - [sumsq_ap[r]] / [sumsq_an[r]] approximate the true real squared
       distances of row [r] of [sa] against [sp] / [sn];
     - [vt[r]] holds the per-row [triplet_step].
   The four distance vectors are bundled into a tuple so we can thread
   a single existential through the loop.  All accesses go through the
   total [sidx] accessor, so no index-range refinement is needed. *)
let carried_pred
  (b : nat) (dd : nat) (#n : nat) (margin eps : f32)
  (sa sp sn : Seq.lseq f32 n)
  (vt : Seq.seq f32)
  (bound : nat)
  (w : tup4 b)
  : prop =
  let (sumsq_ap, sumsq_an, d_ap, d_an) = w in
  forall (r : nat). r < bound ==>
     sidx d_ap r == sqrt (sidx sumsq_ap r) /\
     sidx d_an r == sqrt (sidx sumsq_an r) /\
     sidx sumsq_ap r %~ real_sq_dist eps dd (trow sa dd r) (trow sp dd r) /\
     sidx sumsq_an r %~ real_sq_dist eps dd (trow sa dd r) (trow sn dd r) /\
     sidx vt r == triplet_step margin (sidx d_ap r) (sidx d_an r)

(* Loop-entry witness: vacuously true at [bound = 0]. *)
let triplet_inv_init
  (b dd : nat) (#n : nat) (margin eps : f32)
  (sa sp sn : Seq.lseq f32 n)
  (vt : Seq.seq f32)
  : Lemma (exists (w : tup4 b). carried_pred b dd margin eps sa sp sn vt 0 w)
  = let z : Seq.lseq f32 b = Seq.create b (zero #f32) in
    introduce exists (w : tup4 b). carried_pred b dd margin eps sa sp sn vt 0 w
    with (z, z, z, z)
    and  ()

(* Per-iteration extension: given the freshly computed row-[vi]
   quantities, extend the agreement prefix from [< vi] to [< vi+1]. *)
let memcpy_row_eq
  (#n:nat) (dd:nat) (prev : Seq.lseq f32 dd) (s : Seq.lseq f32 n)
  (vi:nat) (off:nat)
  : Lemma (requires off == vi * dd /\ vi * dd + dd <= n)
          (ensures KS.seq_blit prev 0 s off dd == trow s dd vi)
  = Seq.lemma_eq_intro (KS.seq_blit prev 0 s off dd) (trow s dd vi)

#push-options "--z3rlimit 40"
let triplet_prefix_extend
  (b : nat) (dd : nat) (#n : nat) (margin eps : f32)
  (sa sp sn : Seq.lseq f32 n)
  (vt vt' : Seq.seq f32)
  (vi : nat { vi < b })
  (sumsq_p sumsq_n d_ap_r d_an_r : f32)
  : Lemma
    (requires
      Seq.length vt == b /\ Seq.length vt' == b /\
      vt' == Seq.upd vt vi (triplet_step margin d_ap_r d_an_r) /\
      d_ap_r == sqrt sumsq_p /\
      d_an_r == sqrt sumsq_n /\
      sumsq_p %~ real_sq_dist eps dd (trow sa dd vi) (trow sp dd vi) /\
      sumsq_n %~ real_sq_dist eps dd (trow sa dd vi) (trow sn dd vi) /\
      (exists (w : tup4 b). carried_pred b dd margin eps sa sp sn vt vi w))
    (ensures
      (exists (w : tup4 b). carried_pred b dd margin eps sa sp sn vt' (vi + 1) w))
  = let pw (w : tup4 b) : prop = carried_pred b dd margin eps sa sp sn vt vi w in
    let w0 : (w : tup4 b { pw w }) =
      FStar.IndefiniteDescription.indefinite_description_ghost (tup4 b) pw in
    let (sap, san, dap, dan) = w0 in
    let sap' = Seq.upd sap vi sumsq_p in
    let san' = Seq.upd san vi sumsq_n in
    let dap' = Seq.upd dap vi d_ap_r in
    let dan' = Seq.upd dan vi d_an_r in
    let w' : tup4 b = (sap', san', dap', dan') in
    assert (carried_pred b dd margin eps sa sp sn vt' (vi + 1) w');
    introduce exists (w : tup4 b). carried_pred b dd margin eps sa sp sn vt' (vi + 1) w
    with w'
    and  ()
#pop-options

(* Final discharge: from the loop-exit form (agreement prefix covers
   all [b] rows) and the reduce + scalar-mul outputs, witness
   [triplet_post]. *)
#push-options "--z3rlimit 40"
let triplet_final_lemma
  (b : pos) (dd : nat) (margin eps inv_b : f32)
  (sa sp sn : Seq.lseq f32 (b * dd))
  (vt : Seq.lseq f32 b)
  (s res : f32)
  : Lemma
    (requires
      (exists (w : tup4 b). carried_pred b dd margin eps sa sp sn vt b w) /\
      s %~ rsum (to_real_seq #f32 vt) /\
      res == mul s inv_b)
    (ensures triplet_post b dd margin eps inv_b sa sp sn res)
  = let pw (w : tup4 b) : prop = carried_pred b dd margin eps sa sp sn vt b w in
    let w0 : (w : tup4 b { pw w }) =
      FStar.IndefiniteDescription.indefinite_description_ghost (tup4 b) pw in
    let (sumsq_ap, sumsq_an, d_ap, d_an) = w0 in
    let init_vt : Seq.lseq f32 b =
      Seq.init b (fun r -> triplet_step margin (Seq.index d_ap r) (Seq.index d_an r)) in
    assert (Seq.equal vt init_vt);
    assert (to_real_seq #f32 vt == to_real_seq #f32 init_vt);
    introduce exists (sumsq_ap' sumsq_an' d_ap' d_an' : Seq.lseq f32 b) (s' : f32).
      (forall (r : nat). r < b ==>
         (d_ap' @! r) == sqrt (sumsq_ap' @! r) /\
         (d_an' @! r) == sqrt (sumsq_an' @! r) /\
         (sumsq_ap' @! r) %~ real_sq_dist eps dd (trow sa dd r) (trow sp dd r) /\
         (sumsq_an' @! r) %~ real_sq_dist eps dd (trow sa dd r) (trow sn dd r)) /\
      s' %~ rsum (to_real_seq (Seq.init b (fun r ->
        triplet_step margin (d_ap' @! r) (d_an' @! r)))) /\
      res == mul s' inv_b
    with sumsq_ap sumsq_an d_ap d_an s
    and  ()
#pop-options

(* ── Per-row squared-distance helper. ──────────────────────────────────

   Computes [sumsq = sum_j (x[ri,j] - y[ri,j] + eps)^2] (in f32) for a single
   row [ri] of two flat [len]-element inputs, leaving the result
   pinned (via [%~]) to the genuine real squared distance of the
   corresponding rows.  Factored out so its proof runs in a small,
   clean context (cf. [Kuiper.KB.L2Norm.l2norm_row]). *)
#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn dist_sq_row
  (d : szp { d <= max_blocks * max_threads /\ SZ.fits (d + max_threads) })
  (eps : f32)
  (#len : erased nat)
  (x : array1 f32 (l1_forward len) { is_global x })
  (y : array1 f32 (l1_forward len) { is_global y })
  (scratch_a : array1 f32 (l1_forward d) { is_global scratch_a })
  (scratch_b : array1 f32 (l1_forward d) { is_global scratch_b })
  (off : sz)
  (ri : erased nat)
  (#sx : chest1 f32 len)
  (#sy : chest1 f32 len)
  (#va0 : chest1 f32 d)
  (#vb0 : chest1 f32 d)
  (#fx #fy : perm)
  preserves cpu **
            on gpu_loc (x |-> Frac fx sx) **
            on gpu_loc (y |-> Frac fy sy)
  requires
    on gpu_loc (scratch_a |-> va0) **
    on gpu_loc (scratch_b |-> vb0) **
    pure (SZ.v off == reveal ri * SZ.v d /\
          reveal ri * SZ.v d + SZ.v d <= len)
  returns sumsq : f32
  ensures
    (exists* (va : chest1 f32 d) (vb : chest1 f32 d).
       on gpu_loc (scratch_a |-> va) **
       on gpu_loc (scratch_b |-> vb)) **
    pure (sumsq %~ real_sq_dist eps d
                    (trow (chest1_to_seq (reveal sx)) d (reveal ri))
                    (trow (chest1_to_seq (reveal sy)) d (reveal ri)))
{
  t_memcpy_d2d' scratch_a 0sz x off d;
  with sca. assert (on gpu_loc (scratch_a |-> reveal sca));
  assert pure (chest1_to_seq (reveal sca) ==
               KS.seq_blit (chest1_to_seq (reveal va0)) 0
                 (chest1_to_seq (reveal sx)) off d);
  memcpy_row_eq #len d (chest1_to_seq (reveal va0))
    (chest1_to_seq (reveal sx)) (reveal ri) off;
  assert pure (chest1_to_seq (reveal sca) ==
               trow (chest1_to_seq (reveal sx)) d (reveal ri));

  t_memcpy_d2d' scratch_b 0sz y off d;
  with scb. assert (on gpu_loc (scratch_b |-> reveal scb));
  assert pure (chest1_to_seq (reveal scb) ==
               KS.seq_blit (chest1_to_seq (reveal vb0)) 0
                 (chest1_to_seq (reveal sy)) off d);
  memcpy_row_eq #len d (chest1_to_seq (reveal vb0))
    (chest1_to_seq (reveal sy)) (reveal ri) off;
  assert pure (chest1_to_seq (reveal scb) ==
               trow (chest1_to_seq (reveal sy)) d (reveal ri));

  Map.map_gpu2 #f32 (sq_diff_step_f32 eps) d scratch_a scratch_b;
  with v. assert (on gpu_loc (scratch_a |-> reveal v));
  assert pure (equal (reveal v)
    (Map.chest1_map2 (sq_diff_step_f32 eps) (reveal sca) (reveal scb)));
  Kuiper.Chest.ext (reveal v)
    (Map.chest1_map2 (sq_diff_step_f32 eps) (reveal sca) (reveal scb));
  let vr : chest1 real d =
    hide (seq_to_chest1 (sqdiff_row_real eps d (reveal sca) (reveal scb)));
  sqdiff_map_approx eps d (reveal sca) (reveal scb);
  assert pure (reveal v %~ reveal vr);
  let sumsq = HRed.reduce #f32 id id 1024sz d scratch_a #v vr;
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  chest1_seq_roundtrip (sqdiff_row_real eps d (reveal sca) (reveal scb));
  assert pure (Seq.equal (chest1_to_seq (reveal vr))
                         (sqdiff_row_real eps d (reveal sca) (reveal scb)));
  assert pure (sumsq %~ rsum (sqdiff_row_real eps d (reveal sca) (reveal scb)));
  sqdiff_row_real_eq eps d (reveal sca) (reveal scb)
    (trow (chest1_to_seq (reveal sx)) d (reveal ri))
    (trow (chest1_to_seq (reveal sy)) d (reveal ri));
  real_sq_dist_unfold eps d
    (trow (chest1_to_seq (reveal sx)) d (reveal ri))
    (trow (chest1_to_seq (reveal sy)) d (reveal ri));
  assert pure (sumsq %~ real_sq_dist eps d
                  (trow (chest1_to_seq (reveal sx)) d (reveal ri))
                  (trow (chest1_to_seq (reveal sy)) d (reveal ri)));
  sumsq;
}
#pop-options

#push-options "--z3rlimit 40"
fn triplet_fw_f32
  (b : szp { b <= max_blocks * max_threads /\
             SZ.fits (b + max_threads) })
  (d : szp { d <= max_blocks * max_threads /\
             SZ.fits (d + max_threads) /\
             SZ.fits (b * d) })
  (margin : f32)
  (eps : f32)
  (anchor   : array1 f32 (l1_forward (b * d)) { is_global anchor })
  (positive : array1 f32 (l1_forward (b * d)) { is_global positive })
  (negative : array1 f32 (l1_forward (b * d)) { is_global negative })
  (#sa : chest1 f32 (b * d))
  (#sp : chest1 f32 (b * d))
  (#sn : chest1 f32 (b * d))
  (#fanc #fpos #fneg : perm)
  norewrite
  preserves cpu **
            on gpu_loc (anchor   |-> Frac fanc sa) **
            on gpu_loc (positive |-> Frac fpos sp) **
            on gpu_loc (negative |-> Frac fneg sn)
  returns res : f32
  ensures
    pure (triplet_post b d margin eps (triplet_recip_f32 b)
            (chest1_to_seq sa)
            (chest1_to_seq sp)
            (chest1_to_seq sn)
            res)
{
  let inv_b = triplet_recip_f32 b;

  (* The flat-length nat coincides with [b * d]; rebind the inputs at
     that length so all spec-level uses line up with [triplet_post]. *)
  let sa_c : erased (Seq.lseq f32 (SZ.v b * SZ.v d)) =
    hide (chest1_to_seq (reveal sa) <: Seq.lseq f32 (SZ.v b * SZ.v d));
  let sp_c : erased (Seq.lseq f32 (SZ.v b * SZ.v d)) =
    hide (chest1_to_seq (reveal sp) <: Seq.lseq f32 (SZ.v b * SZ.v d));
  let sn_c : erased (Seq.lseq f32 (SZ.v b * SZ.v d)) =
    hide (chest1_to_seq (reveal sn) <: Seq.lseq f32 (SZ.v b * SZ.v d));

  let scratch_a = alloc0 #f32 d (l1_forward d);
  let scratch_b = alloc0 #f32 d (l1_forward d);
  let t_dev     = alloc0 #f32 b (l1_forward b);
  let t_host    = Vec.alloc #f32 (zero #f32) b;

  triplet_inv_init b d margin eps
    (reveal sa_c) (reveal sp_c) (reveal sn_c)
    (Seq.create b (zero #f32));
  let mut idx : SZ.t = 0sz;
  while (let i = !idx; SZ.(i <^ b))
    invariant
      exists* (vi : sz)
              (vt : Seq.seq f32)
              (va : chest1 f32 d)
              (vb : chest1 f32 d)
              (vt_dev : chest1 f32 b).
        idx |-> vi **
        Vec.pts_to t_host vt **
        on gpu_loc (scratch_a |-> va) **
        on gpu_loc (scratch_b |-> vb) **
        on gpu_loc (t_dev |-> vt_dev) **
        cpu **
        pure (SZ.v vi <= SZ.v b /\
              Seq.length vt == SZ.v b /\
              (exists (w : tup4 b).
                 carried_pred b d margin eps
                   (reveal sa_c) (reveal sp_c) (reveal sn_c)
                   vt vi w))
    decreases (SZ.v b - SZ.v !idx)
  {
    let i = !idx;
    let off : SZ.t = SZ.(i *^ d);
    assert pure (SZ.v off == SZ.v i * SZ.v d);
    FStar.Math.Lemmas.lemma_mult_le_right d (SZ.v i + 1) b;
    assert pure (SZ.v i * SZ.v d + SZ.v d <= SZ.v b * SZ.v d);

    (* ── per-row squared distances via the factored helper ──────── *)
    assert pure (i * d + d <= b * d);

    let sumsq_p = dist_sq_row d eps anchor positive scratch_a scratch_b off (hide (SZ.v i));
    assert pure (trow (chest1_to_seq (reveal sa)) d i == trow (reveal sa_c) d i);
    assert pure (trow (chest1_to_seq (reveal sp)) d i == trow (reveal sp_c) d i);
    assert pure (sumsq_p %~ real_sq_dist eps d
                   (trow (reveal sa_c) d i)
                   (trow (reveal sp_c) d i));
    let d_ap_r = sqrt sumsq_p;

    let sumsq_n = dist_sq_row d eps anchor negative scratch_a scratch_b off (hide (SZ.v i));
    assert pure (trow (chest1_to_seq (reveal sa)) d i == trow (reveal sa_c) d i);
    assert pure (trow (chest1_to_seq (reveal sn)) d i == trow (reveal sn_c) d i);
    assert pure (sumsq_n %~ real_sq_dist eps d
                   (trow (reveal sa_c) d i)
                   (trow (reveal sn_c) d i));
    let d_an_r = sqrt sumsq_n;

    (* ── margin step + store ────────────────────────────────────── *)
    let step = triplet_step_f32 margin d_ap_r d_an_r;
    triplet_step_f32_eq margin d_ap_r d_an_r;

    Vec.pts_to_len t_host;
    Vec.(t_host.(i) <- step);

    with vt_old. assert (Vec.pts_to t_host (reveal vt_old));
    triplet_prefix_extend b d margin eps
      (reveal sa_c) (reveal sp_c) (reveal sn_c)
      (reveal vt_old)
      (Seq.upd (reveal vt_old) i (triplet_step margin d_ap_r d_an_r))
      i sumsq_p sumsq_n d_ap_r d_an_r;

    idx := SZ.(!idx +^ 1sz);
  };

  with vt_loop. assert (Vec.pts_to t_host (reveal vt_loop));
  Vec.pts_to_len t_host;
  t_memcpy_h2d t_dev t_host b;
  with vt_dev_final. assert (on gpu_loc (t_dev |-> reveal vt_dev_final));
  assert pure (chest1_to_seq (reveal vt_dev_final) == reveal vt_loop);

  let vt_r : chest1 real b = hide (to_real_chest (reveal vt_dev_final));
  lemma_to_real_chest_approximates (reveal vt_dev_final);
  let s = HRed.reduce #f32 id id 1024sz b t_dev vt_r;
  assert pure (equal (chest_map id (reveal vt_r)) (reveal vt_r));
  lem_to_real_chest_to_seq (reveal vt_dev_final);
  assert pure (s %~ rsum (to_real_seq #f32 (reveal vt_loop)));

  let m : f32 = mul s inv_b;

  triplet_final_lemma b d margin eps inv_b
    (reveal sa_c) (reveal sp_c) (reveal sn_c)
    (reveal vt_loop <: Seq.lseq f32 b)
    s
    m;

  Vec.free t_host;
  free scratch_a;
  free scratch_b;
  free t_dev;
  m;
}
#pop-options
