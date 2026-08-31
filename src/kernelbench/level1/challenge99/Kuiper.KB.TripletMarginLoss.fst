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
module SqrtApprox = Kuiper.KB.Compat.SqrtApprox
module RealSqrt = FStar.Math.Sqrt
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

(* ── Approximation lemmas from the f32 implementation to the direct
      real-valued specification. ─────────────────────────────────── *)

let sqdiff_approx (eps x y : f32) (rx ry : real)
  : Lemma (requires x %~ rx /\ y %~ ry)
          (ensures sq_diff_step_f32 eps x y
                   %~ sqdiff_step_r (to_real eps) rx ry)
  = to_real_ok eps;
    sub_approx x y rx ry;
    a_add (sub x y) eps (rx -. ry) (to_real eps);
    a_mul (add (sub x y) eps) (add (sub x y) eps)
      ((rx -. ry) +. to_real eps) ((rx -. ry) +. to_real eps)

let triplet_step_approx
  (margin d_ap d_an : f32)
  (rd_ap rd_an : real)
  : Lemma (requires d_ap %~ rd_ap /\ d_an %~ rd_an)
          (ensures triplet_step_f32 margin d_ap d_an
                   %~ real_triplet_step (to_real margin) rd_ap rd_an)
  = to_real_ok margin;
    to_real_ok (zero #f32);
    sub_approx d_ap d_an rd_ap rd_an;
    a_add (sub d_ap d_an) margin (rd_ap -. rd_an) (to_real margin);
    fmax_approx (zero #f32) (add (sub d_ap d_an) margin)
      0.0R ((rd_ap -. rd_an) +. to_real margin)

(* A valid row slice of related flat sequences remains related. *)
let row_slice_approx
  (#n:nat)
  (sf : Seq.lseq f32 n)
  (sr : Seq.lseq real n)
  (d r : nat)
  : Lemma
      (requires sf %~ sr /\ r * d + d <= n)
      (ensures Seq.slice sf (r * d) (r * d + d) %~ trow sr d r)
  = introduce forall (i:nat{i < d}).
      Seq.index (Seq.slice sf (r * d) (r * d + d)) i
        %~ Seq.index (trow sr d r) i
    with ()

(* Turn a sequence approximation into the corresponding chest
   approximation after a memcpy-established sequence equality. *)
let chest_from_seq_approx
  (#n:nat)
  (c : chest1 f32 n)
  (sf : Seq.lseq f32 n)
  (sr : Seq.lseq real n)
  : Lemma
      (requires chest1_to_seq c == sf /\ sf %~ sr)
      (ensures c %~ seq_to_chest1 sr)
  = let aux (i:natlt n)
      : Lemma (acc1 c i %~ acc1 (seq_to_chest1 sr) i)
      = ()
    in
    Classical.forall_intro aux

let chest_to_seq_approx
  (#n:nat)
  (c : chest1 f32 n)
  (sr : Seq.lseq real n)
  : Lemma
      (requires c %~ seq_to_chest1 sr)
      (ensures chest1_to_seq c %~ sr)
  = introduce forall (i:nat{i < n}).
      (chest1_to_seq c @! i) %~ (sr @! i)
    with ()

#push-options ""
let sqdiff_map_approx
  (eps : f32)
  (d:nat)
  (sca scb : chest1 f32 d)
  (ra rb : Seq.lseq real d)
  : Lemma
      (requires sca %~ seq_to_chest1 ra /\ scb %~ seq_to_chest1 rb)
      (ensures Map.chest1_map2 (sq_diff_step_f32 eps) sca scb
               %~ seq_to_chest1 (Seq.init d (fun j ->
                    sqdiff_step_r (to_real eps) (ra @! j) (rb @! j))))
  = let aux (i:natlt d)
      : Lemma
          (acc1 (Map.chest1_map2 (sq_diff_step_f32 eps) sca scb) i
           %~ acc1 (seq_to_chest1 (Seq.init d (fun j ->
                sqdiff_step_r (to_real eps) (ra @! j) (rb @! j)))) i)
      = sqdiff_approx eps (acc1 sca i) (acc1 scb i) (ra @! i) (rb @! i)
    in
    Classical.forall_intro aux
#pop-options

let real_sq_dist_unfold
  (eps : real)
  (d : nat)
  (ra rb : Seq.lseq real d)
  : Lemma
      (real_sq_dist eps d ra rb ==
       rsum (Seq.init d (fun j -> sqdiff_step_r eps (ra @! j) (rb @! j))))
  = ()

noextract
let sidx (s : Seq.seq f32) (r : nat) : f32 =
  if r < Seq.length s then Seq.index s r else zero

(* The loop carries only the public real-valued fact: every completed
   host-vector element approximates its ideal triplet-loss term. *)
let carried_pred
  (b : nat)
  (terms : Seq.lseq real b)
  (vt : Seq.seq f32)
  (bound : nat)
  : prop =
  bound <= b /\ Seq.length vt == b /\
  (forall (r : nat). r < bound ==> sidx vt r %~ (terms @! r))

#push-options "--z3rlimit 40"
let triplet_prefix_extend
  (b : nat)
  (terms : Seq.lseq real b)
  (vt vt' : Seq.seq f32)
  (vi : nat { vi < b })
  (step : f32)
  : Lemma
      (requires carried_pred b terms vt vi /\
                vt' == Seq.upd vt vi step /\
                step %~ (terms @! vi))
      (ensures carried_pred b terms vt' (vi + 1))
  = ()
#pop-options

let carried_complete
  (b : nat)
  (terms : Seq.lseq real b)
  (vt : Seq.seq f32)
  : Lemma (requires carried_pred b terms vt b)
          (ensures (vt <: Seq.lseq f32 b) %~ terms)
  = introduce forall (i:nat{i < b}).
      ((vt <: Seq.lseq f32 b) @! i) %~ (terms @! i)
    with ()

let memcpy_row_eq
  (#n:nat) (dd:nat) (prev : Seq.lseq f32 dd) (s : Seq.lseq f32 n)
  (vi:nat) (off:nat)
  : Lemma (requires off == vi * dd /\ vi * dd + dd <= n)
          (ensures KS.seq_blit prev 0 s off dd ==
                   Seq.slice s (vi * dd) (vi * dd + dd))
  = Seq.lemma_eq_intro (KS.seq_blit prev 0 s off dd)
      (Seq.slice s (vi * dd) (vi * dd + dd))

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
  (rx : erased (Seq.lseq real len))
  (ry : erased (Seq.lseq real len))
  (#sx : chest1 f32 len)
  (#sy : chest1 f32 len)
  (#va0 : chest1 f32 d)
  (#vb0 : chest1 f32 d)
  (#fx #fy : perm)
  preserves cpu **
            on gpu_loc (x |-> Frac fx sx) **
            on gpu_loc (y |-> Frac fy sy) **
            pure (sx %~ seq_to_chest1 rx /\ sy %~ seq_to_chest1 ry)
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
    pure (sumsq %~ real_sq_dist (to_real eps) d
                    (trow rx d (reveal ri))
                    (trow ry d (reveal ri)))
{
  t_memcpy_d2d' scratch_a 0sz x off d;
  with sca. assert (on gpu_loc (scratch_a |-> reveal sca));
  assert pure (chest1_to_seq (reveal sca) ==
               KS.seq_blit (chest1_to_seq (reveal va0)) 0
                 (chest1_to_seq (reveal sx)) off d);
  memcpy_row_eq #len d (chest1_to_seq (reveal va0))
    (chest1_to_seq (reveal sx)) (reveal ri) off;
  assert pure (chest1_to_seq (reveal sca) ==
               Seq.slice (chest1_to_seq (reveal sx))
                 (reveal ri * d) (reveal ri * d + d));
  chest_to_seq_approx (reveal sx) rx;
  row_slice_approx (chest1_to_seq (reveal sx)) rx d (reveal ri);
  chest_from_seq_approx (reveal sca)
    (Seq.slice (chest1_to_seq (reveal sx))
      (reveal ri * d) (reveal ri * d + d))
    (trow rx d (reveal ri));

  t_memcpy_d2d' scratch_b 0sz y off d;
  with scb. assert (on gpu_loc (scratch_b |-> reveal scb));
  assert pure (chest1_to_seq (reveal scb) ==
               KS.seq_blit (chest1_to_seq (reveal vb0)) 0
                 (chest1_to_seq (reveal sy)) off d);
  memcpy_row_eq #len d (chest1_to_seq (reveal vb0))
    (chest1_to_seq (reveal sy)) (reveal ri) off;
  assert pure (chest1_to_seq (reveal scb) ==
               Seq.slice (chest1_to_seq (reveal sy))
                 (reveal ri * d) (reveal ri * d + d));
  chest_to_seq_approx (reveal sy) ry;
  row_slice_approx (chest1_to_seq (reveal sy)) ry d (reveal ri);
  chest_from_seq_approx (reveal scb)
    (Seq.slice (chest1_to_seq (reveal sy))
      (reveal ri * d) (reveal ri * d + d))
    (trow ry d (reveal ri));

  Map.map_gpu2 #f32 (sq_diff_step_f32 eps) d scratch_a scratch_b;
  with v. assert (on gpu_loc (scratch_a |-> reveal v));
  assert pure (equal (reveal v)
    (Map.chest1_map2 (sq_diff_step_f32 eps) (reveal sca) (reveal scb)));
  Kuiper.Chest.ext (reveal v)
    (Map.chest1_map2 (sq_diff_step_f32 eps) (reveal sca) (reveal scb));
  let terms : erased (Seq.lseq real d) =
    hide (Seq.init d (fun j -> sqdiff_step_r (to_real eps)
      (trow rx d (reveal ri) @! j)
      (trow ry d (reveal ri) @! j)));
  let vr : chest1 real d = hide (seq_to_chest1 (reveal terms));
  sqdiff_map_approx eps d (reveal sca) (reveal scb)
    (trow rx d (reveal ri)) (trow ry d (reveal ri));
  assert pure (reveal v %~ reveal vr);
  let sumsq = HRed.reduce #f32 id id 1024sz d scratch_a #v vr;
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  chest1_seq_roundtrip (reveal terms);
  assert pure (Seq.equal (chest1_to_seq (reveal vr)) (reveal terms));
  assert pure (sumsq %~ rsum (reveal terms));
  real_sq_dist_unfold (to_real eps) d
    (trow rx d (reveal ri)) (trow ry d (reveal ri));
  assert pure (sumsq %~ real_sq_dist (to_real eps) d
                  (trow rx d (reveal ri))
                  (trow ry d (reveal ri)));
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
  (ra : erased (Seq.lseq real (b * d)))
  (rp : erased (Seq.lseq real (b * d)))
  (rn : erased (Seq.lseq real (b * d)))
  (#fanc #fpos #fneg : perm)
  norewrite
  preserves cpu **
            on gpu_loc (anchor   |-> Frac fanc sa) **
            on gpu_loc (positive |-> Frac fpos sp) **
            on gpu_loc (negative |-> Frac fneg sn) **
            pure (sa %~ seq_to_chest1 ra /\
                  sp %~ seq_to_chest1 rp /\
                  sn %~ seq_to_chest1 rn)
  returns res : f32
  ensures
    pure (triplet_post b d margin eps ra rp rn res)
{
  let inv_b = triplet_recip_f32 b;
  let terms : erased (Seq.lseq real b) =
    hide (real_triplet_terms b d (to_real margin) (to_real eps) ra rp rn);

  let scratch_a = alloc0 #f32 d (l1_forward d);
  let scratch_b = alloc0 #f32 d (l1_forward d);
  let t_dev     = alloc0 #f32 b (l1_forward b);
  let t_host    = Vec.alloc #f32 (zero #f32) b;

  assert pure (carried_pred b (reveal terms)
    (Seq.create b (zero #f32)) 0);
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
              carried_pred b (reveal terms) vt vi)
    decreases (SZ.v b - SZ.v !idx)
  {
    let i = !idx;
    let off : SZ.t = SZ.(i *^ d);
    assert pure (SZ.v off == SZ.v i * SZ.v d);
    FStar.Math.Lemmas.lemma_mult_le_right d (SZ.v i + 1) b;
    assert pure (SZ.v i * SZ.v d + SZ.v d <= SZ.v b * SZ.v d);

    (* ── per-row squared distances via the factored helper ──────── *)
    assert pure (i * d + d <= b * d);

    let sumsq_p = dist_sq_row d eps anchor positive scratch_a scratch_b off
      (hide (SZ.v i)) ra rp;
    let rsq_ap = real_sq_dist (to_real eps) d (trow ra d i) (trow rp d i);
    real_sq_dist_nonnegative (to_real eps) d (trow ra d i) (trow rp d i);
    SqrtApprox.sqrt_approx sumsq_p rsq_ap;
    let d_ap_r = sqrt sumsq_p;

    let sumsq_n = dist_sq_row d eps anchor negative scratch_a scratch_b off
      (hide (SZ.v i)) ra rn;
    let rsq_an = real_sq_dist (to_real eps) d (trow ra d i) (trow rn d i);
    real_sq_dist_nonnegative (to_real eps) d (trow ra d i) (trow rn d i);
    SqrtApprox.sqrt_approx sumsq_n rsq_an;
    let d_an_r = sqrt sumsq_n;

    (* ── margin step + store ────────────────────────────────────── *)
    let step = triplet_step_f32 margin d_ap_r d_an_r;
    triplet_step_approx margin d_ap_r d_an_r
      (RealSqrt.sqrt rsq_ap) (RealSqrt.sqrt rsq_an);
    assert pure (step %~ (reveal terms @! i));

    Vec.pts_to_len t_host;
    Vec.(t_host.(i) <- step);

    with vt_old. assert (Vec.pts_to t_host (reveal vt_old));
    triplet_prefix_extend b (reveal terms)
      (reveal vt_old)
      (Seq.upd (reveal vt_old) i step)
      i step;

    idx := SZ.(!idx +^ 1sz);
  };

  with vt_loop. assert (Vec.pts_to t_host (reveal vt_loop));
  Vec.pts_to_len t_host;
  t_memcpy_h2d t_dev t_host b;
  with vt_dev_final. assert (on gpu_loc (t_dev |-> reveal vt_dev_final));
  assert pure (chest1_to_seq (reveal vt_dev_final) == reveal vt_loop);

  carried_complete b (reveal terms) (reveal vt_loop);
  chest_from_seq_approx (reveal vt_dev_final)
    (reveal vt_loop <: Seq.lseq f32 b) (reveal terms);
  let vt_r : chest1 real b = hide (seq_to_chest1 (reveal terms));
  let s = HRed.reduce #f32 id id 1024sz b t_dev vt_r;
  assert pure (equal (chest_map id (reveal vt_r)) (reveal vt_r));
  chest1_seq_roundtrip (reveal terms);
  assert pure (s %~ rsum (reveal terms));

  let b64 : Int64.t = FStar.Int.Cast.uint64_to_int64
    (FStar.SizeT.sizet_to_uint64 b);
  assert pure (Int64.v b64 == SZ.v b);
  let bf : f32 = of_int b64;
  of_int_approx #f32 b64;
  assert pure (bf %~ Real.of_int b);
  assert pure (inv_b == div one bf);
  div_approx (one #f32) bf 1.0R (Real.of_int b);
  let m : f32 = mul s inv_b;
  a_mul s inv_b (rsum (reveal terms)) (1.0R /. Real.of_int b);
  assert pure (rsum (reveal terms) == rsum (real_triplet_terms b d
    (to_real margin) (to_real eps) ra rp rn));
  real_triplet_loss_mul b d (to_real margin) (to_real eps) ra rp rn;
  assert pure (m %~ real_triplet_loss b d
    (to_real margin) (to_real eps) ra rp rn);

  Vec.free t_host;
  free scratch_a;
  free scratch_b;
  free t_dev;
  m;
}
#pop-options
