module Kuiper.KB.CrossEntropyLoss

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Approximates
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.CrossEntropyLoss
open Kuiper.Seq.Common { (@!) }
module SZ = Kuiper.SizeT
module LSM = Kuiper.Kernel.LogSoftmax
module HRed = Kuiper.Kernel.HReduce
module Vec = Pulse.Lib.Vec
module KS = Kuiper.Seq.Common

(* Verified, extractable reciprocal 1/B as f32 (extracts to
   1.0f / (float)(int64_t)(uint64_t)b), so the mean's 1/B factor is
   computed inside the verification boundary. *)
let ce_recip_f32 (b : szp) : f32 =
  div one (of_int (FStar.Int.Cast.uint64_to_int64
                     (FStar.SizeT.sizet_to_uint64 b)))

let seq_map_id_eq (#a:Type) (s : Seq.seq a)
  : Lemma (Seq.equal (KS.seq_map id s) s)
  = ()

(* ─────────────────────────────────────────────────────────────────────
   chest/seq bridge lemmas + tensor-level device memcpy / single-read
   helpers.  These wrap the core [larray] [gpu_memcpy_*] primitives with
   the [tensor_concr]/[tensor_abs] bridge (cf. Kuiper.Kernel.LogSoftmax),
   exposing postconditions in [chest1_to_seq] / [acc1] terms.
   ───────────────────────────────────────────────────────────────────── *)
let lem_to_seq (#et:Type) (n:nat) (c : chest1 et n)
  : Lemma (to_seq (l1_forward n) c == chest1_to_seq c)
  = assert (Seq.equal (to_seq (l1_forward n) c) (chest1_to_seq c))

let lem_index_chest1 (#et:Type) (#n:nat) (c : chest1 et n) (i:natlt n)
  : Lemma (Seq.index (chest1_to_seq c) i == acc1 c i)
  = ()

let lem_blit_index_0 (#a:Type) (prev : Seq.seq a) (src : Seq.seq a) (i:nat)
  : Lemma (requires Seq.length prev == 1 /\ i < Seq.length src)
          (ensures Seq.index (KS.seq_blit prev 0 src i 1) 0 == Seq.index src i)
  = ()

let lem_to_real_chest_to_seq (#et:Type0) {| scalar et, real_like et |} (#n:nat)
  (c : chest1 et n)
  : Lemma (chest1_to_seq (to_real_chest c) == to_real_seq (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (to_real_chest c)) (to_real_seq (chest1_to_seq c)))

(* forward direction: a chest approximation gives per-index approximation *)
let approx_at (#et:Type0) {| scalar et, real_like et |} (#n:nat)
  (c1 : chest1 et n) (c2 : chest1 real n) (i:natlt n)
  : Lemma (requires c1 %~ c2) (ensures acc1 c1 i %~ acc1 c2 i)
  = ()

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
  (#v : erased (chest1 a src_sz))
  (#gv : erased (chest1 a dst_sz))
  preserves cpu ** on gpu_loc (src |-> Frac f v)
  requires on gpu_loc (dst |-> gv)
  ensures exists* (s' : chest1 a dst_sz).
      on gpu_loc (dst |-> s') **
      pure (chest1_to_seq s' ==
            KS.seq_blit (chest1_to_seq gv) (SZ.v dst_off) (chest1_to_seq v) (SZ.v src_off) (SZ.v cnt))
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
  (#gv : erased (chest1 a sz))
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

(* single-element host read of a device array *)
inline_for_extraction noextract
fn t_read_1
  (#a:Type u#0) {| sized a |}
  (#sz : erased nat)
  (arr : array1 a (l1_forward sz))
  (i : SZ.t { SZ.v i < sz })
  (dummy : a)
  (#f : perm)
  (#va : erased (chest1 a sz))
  preserves cpu ** on gpu_loc (arr |-> Frac f va)
  returns x : a
  ensures pure (x == acc1 va (SZ.v i))
{
  let tmp = Vec.alloc dummy 1sz;
  Vec.pts_to_len tmp;
  map_loc gpu_loc #(arr |-> Frac f va) #(core arr |-> Frac f (to_seq (l1_forward sz) va))
    fn _ { tensor_concr arr; };
  gpu_memcpy_device_to_host'
    #a #_ #(hide 1) tmp 0sz (core arr) i 1sz;
  with s'. assert (Vec.pts_to tmp s');
  map_loc gpu_loc #(core arr |-> Frac f (to_seq (l1_forward sz) va)) #(arr |-> Frac f va)
    fn _ {
      tensor_abs (l1_forward sz) (core arr);
      rewrite (from_array (l1_forward sz) (core arr) |-> Frac f va)
           as (arr |-> Frac f va);
    };
  Vec.pts_to_len tmp;
  let x = Vec.(tmp.(0sz));
  Vec.free tmp;
  lem_to_seq sz va;
  lem_blit_index_0 (Seq.create 1 dummy) (to_seq (l1_forward sz) va) (SZ.v i);
  lem_index_chest1 va (SZ.v i);
  x
}

(* IEEE negation [sub zero v] approximates the real negation, proven in
   a small clean context to avoid solver blowups in the kernel body. *)
let neg_approx_f32 (v : f32) (rv : real)
  : Lemma (requires v %~ rv)
          (ensures (sub (zero #f32) v) %~ (0.0R -. rv))
  = assert ((zero #f32) %~ 0.0R);
    sub_approx #f32 (zero #f32) v 0.0R rv

(* Unfold [ce_term_r] for an in-range target: the guard reduces to the
   real log-softmax branch. *)
let ce_term_fold (c : pos) (#n : nat) (sp : Seq.lseq f32 n) (r : nat) (t : nat)
  : Lemma (requires t < c)
          (ensures ce_term_r c sp r t ==
                   (0.0R -. acc1 (LSM.log_softmax_real
                              (seq_to_chest1 (to_real_seq (crow sp c r) <: Seq.lseq real c))) t))
  = ()

(* The real chest fed to [log_softmax_gpu] (the [to_real] image of the
   freshly copied row [sca]) is exactly the spec's per-row real chest. *)
let ce_ra_eq (c : pos) (#n : nat) (sp_s : Seq.lseq f32 n) (r : nat) (sca : chest1 f32 c)
  : Lemma (requires chest1_to_seq sca == crow sp_s c r)
          (ensures to_real_chest sca ==
                   seq_to_chest1 (to_real_seq (crow sp_s c r) <: Seq.lseq real c))
  = assert (Kuiper.Chest.equal (to_real_chest sca)
              (seq_to_chest1 (to_real_seq (crow sp_s c r) <: Seq.lseq real c)))

(* The device-to-device row copy lands exactly row [r] (= [crow]). *)
let memcpy_row_eq
  (#n:nat) (c:nat) (prev : Seq.lseq f32 c) (s : Seq.lseq f32 n)
  (r:nat) (off:nat)
  : Lemma (requires off == r * c /\ r * c + c <= n)
          (ensures KS.seq_blit prev 0 s off c == crow s c r)
  = Seq.lemma_eq_intro (KS.seq_blit prev 0 s off c) (crow s c r)

(* [crow] depends only on the underlying [Seq.seq] and the numeric
   length, so it is stable under a length-coercion of the buffer. *)
let crow_coerce
  (#n1 #n2:nat) (s1 : Seq.lseq f32 n1) (s2 : Seq.lseq f32 n2) (c r:nat)
  : Lemma (requires n1 == n2 /\ (s1 <: Seq.seq f32) == (s2 <: Seq.seq f32))
          (ensures crow s1 c r == crow s2 c r)
  = ()

(* Total accessor for the (host) per-row loss vector. *)
let sidx (s : Seq.seq f32) (r : nat) : f32 =
  if r < Seq.length s then Seq.index s r else zero

(* Total accessor for the target index of row [r] (as a nat). *)
let tgt (#b : nat) (stv : Seq.lseq SZ.t b) (r : nat) : nat =
  if r < b then SZ.v (stv @! r) else 0

(* For every already-processed row [r < bound], the stored per-row loss
   approximates the genuine real CE term of row [r] at its target. *)
let ce_carried
  (b : nat) (c : pos) (#n : nat)
  (sp : Seq.lseq f32 n) (stv : Seq.lseq SZ.t b)
  (vt : Seq.seq f32) (bound : nat)
  : prop =
  forall (r : nat). r < bound ==>
    sidx vt r %~ ce_term_r c sp r (tgt stv r)

(* Loop-entry witness (vacuous at [bound = 0]). *)
let ce_inv_init
  (b : nat) (c : pos) (#n : nat)
  (sp : Seq.lseq f32 n) (stv : Seq.lseq SZ.t b)
  (vt : Seq.seq f32)
  : Lemma (ce_carried b c sp stv vt 0)
  = ()

(* Per-iteration extension: storing the row-[vi] loss into slot [vi]
   extends the agreement prefix from [< vi] to [< vi+1]. *)
let ce_prefix_extend
  (b : nat) (c : pos) (#n : nat)
  (sp : Seq.lseq f32 n) (stv : Seq.lseq SZ.t b)
  (vt vt' : Seq.seq f32) (vi : nat { vi < b }) (newv : f32)
  : Lemma
    (requires
      Seq.length vt == b /\ Seq.length vt' == b /\
      vt' == Seq.upd vt vi newv /\
      newv %~ ce_term_r c sp vi (tgt stv vi) /\
      ce_carried b c sp stv vt vi)
    (ensures ce_carried b c sp stv vt' (vi + 1))
  = ()

(* Final discharge: from the full agreement prefix and the
   reduce + scalar-mul outputs, witness [cross_entropy_post]. *)
let ce_final_lemma
  (b : pos) (c : pos)
  (inv_b : f32)
  (sp : Seq.lseq f32 (b * c)) (stv : Seq.lseq SZ.t b)
  (vt : Seq.lseq f32 b)
  (s res : f32)
  : Lemma
    (requires
      ce_carried b c sp stv vt b /\
      s %~ rsum (to_real_seq vt) /\
      res == mul s inv_b)
    (ensures cross_entropy_post b c inv_b sp stv res)
  = introduce exists (per_batch : Seq.lseq f32 b) (s' : f32).
      (forall (r : nat). r < b ==>
         (per_batch @! r) %~ ce_term_r c sp r (SZ.v (stv @! r))) /\
      s' %~ rsum (to_real_seq per_batch) /\
      res == mul s' inv_b
    with vt s
    and  ()

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn ce_loss_impl
  (b : szp { b <= max_blocks * max_threads /\
             SZ.fits (b + max_threads) })
  (c : szp { c <= max_blocks * max_threads /\
             SZ.fits (c + max_threads) /\
             SZ.fits (b * c) })
  (inv_b : f32)
  (predictions : array1 f32 (l1_forward (b *^ c)) { is_global predictions })
  (targets : array1 SZ.t (l1_forward b) { is_global targets })
  (#sp : erased (chest1 f32 (b *^ c)))
  (#stv : erased (chest1 SZ.t b))
  (#fp #ft : perm)
  norewrite
  preserves cpu **
            on gpu_loc (predictions |-> Frac fp sp) **
            on gpu_loc (targets |-> Frac ft stv)
  requires
    pure (forall (r : nat). r < SZ.v b ==> SZ.v (acc1 (reveal stv) r) < SZ.v c)
  returns res : f32
  ensures
    pure (cross_entropy_post (SZ.v b) (SZ.v c) inv_b
            (chest1_to_seq (reveal sp) <: Seq.lseq f32 (SZ.v b * SZ.v c))
            (chest1_to_seq (reveal stv) <: Seq.lseq SZ.t (SZ.v b))
            res)
{
  assert pure (SZ.v (b *^ c) == SZ.v b * SZ.v c);
  let sp_c : erased (Seq.lseq f32 (SZ.v b * SZ.v c)) =
    hide (chest1_to_seq (reveal sp) <: Seq.lseq f32 (SZ.v b * SZ.v c));
  let stv_s : erased (Seq.lseq SZ.t (SZ.v b)) =
    hide (chest1_to_seq (reveal stv));
  assert pure ((reveal sp_c <: Seq.seq f32) == (chest1_to_seq (reveal sp) <: Seq.seq f32));

  let scratch = alloc0 #f32 c (l1_forward c);
  let t_dev   = alloc0 #f32 b (l1_forward b);
  let t_host  = Vec.alloc #f32 (zero #f32) b;

  ce_inv_init (SZ.v b) (SZ.v c) (reveal sp_c) (reveal stv_s)
    (Seq.create (SZ.v b) (zero #f32));
  let mut idx : SZ.t = 0sz;
  while (let i = !idx; SZ.(i <^ b))
    invariant
      exists* (vi : sz)
              (vt : Seq.seq f32)
              (va : chest1 f32 c)
              (vt_dev : chest1 f32 b).
        idx |-> vi **
        Vec.pts_to t_host vt **
        on gpu_loc (scratch |-> va) **
        on gpu_loc (t_dev |-> vt_dev) **
        cpu **
        pure (SZ.v vi <= SZ.v b /\
              Seq.length vt == SZ.v b /\
              ce_carried (SZ.v b) (SZ.v c) (reveal sp_c) (reveal stv_s) vt (SZ.v vi))
    decreases (SZ.v b - SZ.v !idx)
  {
    let i = !idx;
    let off : SZ.t = SZ.(i *^ c);
    assert pure (SZ.v off == SZ.v i * SZ.v c);
    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v c) (SZ.v i + 1) (SZ.v b);
    assert pure (SZ.v i * SZ.v c + SZ.v c <= SZ.v b * SZ.v c);
    assert pure (SZ.v i * SZ.v c + SZ.v c <= SZ.v (b *^ c));

    (* ── copy row [i] into scratch ──────────────────────────────── *)
    with va_prev. assert (on gpu_loc (scratch |-> reveal va_prev));
    t_memcpy_d2d' scratch 0sz predictions off c;
    with sca. assert (on gpu_loc (scratch |-> reveal sca));
    assert pure (chest1_to_seq (reveal sca) ==
                 KS.seq_blit (chest1_to_seq (reveal va_prev)) 0
                             (chest1_to_seq (reveal sp)) (SZ.v off) (SZ.v c));
    memcpy_row_eq #(SZ.v (b *^ c)) (SZ.v c)
      (chest1_to_seq (reveal va_prev)) (chest1_to_seq (reveal sp)) (SZ.v i) (SZ.v off);
    assert pure (chest1_to_seq (reveal sca) == crow (chest1_to_seq (reveal sp)) (SZ.v c) (SZ.v i));
    crow_coerce #(SZ.v (b *^ c)) #(SZ.v b * SZ.v c)
      (chest1_to_seq (reveal sp)) (reveal sp_c) (SZ.v c) (SZ.v i);
    assert pure (chest1_to_seq (reveal sca) == crow (reveal sp_c) (SZ.v c) (SZ.v i));

    (* ── verified numerically-stable log-softmax in place ───────── *)
    let ra : erased (chest1 real (SZ.v c)) = hide (to_real_chest (reveal sca));
    lemma_to_real_chest_approximates (reveal sca);
    assert pure (reveal sca %~ reveal ra);
    LSM.log_softmax_gpu #f32 1024sz scratch ra;
    with sca'. assert (on gpu_loc (scratch |-> reveal sca'));
    assert pure (reveal sca' %~ LSM.log_softmax_real (reveal ra));
    ce_ra_eq (SZ.v c) (reveal sp_c) (SZ.v i) (reveal sca);
    assert pure (reveal ra ==
                 seq_to_chest1 (to_real_seq (crow (reveal sp_c) (SZ.v c) (SZ.v i)) <: Seq.lseq real (SZ.v c)));

    (* ── gather the (negated) target lane ───────────────────────── *)
    let ti = t_read_1 targets i 0sz;
    lem_index_chest1 (reveal stv) (SZ.v i);
    assert pure (ti == acc1 (reveal stv) (SZ.v i));
    assert pure (SZ.v ti < SZ.v c);
    let v = t_read_1 scratch ti (zero #f32);
    assert pure (v == acc1 (reveal sca') (SZ.v ti));
    (* pointwise: scratch'[ti] approximates log_softmax(...)[ti] *)
    approx_at (reveal sca') (LSM.log_softmax_real (reveal ra)) (SZ.v ti);
    assert pure (v %~ acc1 (LSM.log_softmax_real (reveal ra)) (SZ.v ti));
    let neg_v : f32 = sub (zero #f32) v;
    neg_approx_f32 v (acc1 (LSM.log_softmax_real (reveal ra)) (SZ.v ti));
    (* fold back to the spec term [ce_term_r] *)
    ce_term_fold (SZ.v c) (reveal sp_c) (SZ.v i) (SZ.v ti);
    assert pure (neg_v %~ ce_term_r (SZ.v c) (reveal sp_c) (SZ.v i) (SZ.v ti));
    assert pure (tgt (reveal stv_s) (SZ.v i) == SZ.v ti);
    assert pure (neg_v %~ ce_term_r (SZ.v c) (reveal sp_c) (SZ.v i) (tgt (reveal stv_s) (SZ.v i)));

    (* ── store per-row loss ─────────────────────────────────────── *)
    Vec.pts_to_len t_host;
    Vec.(t_host.(i) <- neg_v);
    with vt_old. assert (Vec.pts_to t_host (reveal vt_old));
    ce_prefix_extend (SZ.v b) (SZ.v c) (reveal sp_c) (reveal stv_s)
      (reveal vt_old) (Seq.upd (reveal vt_old) (SZ.v i) neg_v)
      (SZ.v i) neg_v;

    idx := SZ.(!idx +^ 1sz);
  };

  (* ── on-device reduce-sum of the B per-row losses ───────────────── *)
  with vt_loop. assert (Vec.pts_to t_host (reveal vt_loop));
  Vec.pts_to_len t_host;
  t_memcpy_h2d t_dev t_host b;
  with vt_dev_final. assert (on gpu_loc (t_dev |-> reveal vt_dev_final));
  assert pure (chest1_to_seq (reveal vt_dev_final) == reveal vt_loop);

  let vr : erased (chest1 real (SZ.v b)) = hide (to_real_chest (reveal vt_dev_final));
  lemma_to_real_chest_approximates (reveal vt_dev_final);
  let s = HRed.reduce #f32 id id 1024sz b t_dev vr;
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  lem_to_real_chest_to_seq (reveal vt_dev_final);
  assert pure (s %~ rsum (to_real_seq #f32 (reveal vt_loop)));

  let m : f32 = mul s inv_b;

  ce_final_lemma (SZ.v b) (SZ.v c) inv_b
    (reveal sp_c) (reveal stv_s)
    (reveal vt_loop <: Seq.lseq f32 (SZ.v b))
    s m;

  Vec.free t_host;
  free scratch;
  free t_dev;
  m;
}
#pop-options

#push-options "--z3rlimit 40"
let ce_loss_fw_f32 : ce_loss_fw_ty =
  fun b c inv_b predictions targets #sp #stv #fp #ft ->
    ce_loss_impl b c inv_b predictions targets #sp #stv #fp #ft
#pop-options
