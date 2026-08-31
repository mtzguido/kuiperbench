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

(* Verified reciprocal 1/B, inlined into the public entry point. *)
inline_for_extraction noextract
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

let chest_to_seq_approx
  (#n:nat) (c : chest1 f32 n) (sr : Seq.lseq real n)
  : Lemma (requires c %~ seq_to_chest1 sr)
          (ensures chest1_to_seq c %~ sr)
  = introduce forall (i:nat{i < n}).
      (chest1_to_seq c @! i) %~ (sr @! i)
    with ()

let row_slice_approx
  (#n:nat) (sf : Seq.lseq f32 n) (sr : Seq.lseq real n) (c r:nat)
  : Lemma (requires sf %~ sr /\ r * c + c <= n)
          (ensures Seq.slice sf (r * c) (r * c + c) %~ crow sr c r)
  = introduce forall (i:nat{i < c}).
      Seq.index (Seq.slice sf (r * c) (r * c + c)) i
        %~ Seq.index (crow sr c r) i
    with ()

let chest_from_seq_approx
  (#n:nat) (c : chest1 f32 n) (sf : Seq.lseq f32 n)
  (sr : Seq.lseq real n)
  : Lemma (requires chest1_to_seq c == sf /\ sf %~ sr)
          (ensures c %~ seq_to_chest1 sr)
  = let aux (i:natlt n)
      : Lemma (acc1 c i %~ acc1 (seq_to_chest1 sr) i)
      = ()
    in
    Classical.forall_intro aux

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

(* single-element host read of a device array *)
inline_for_extraction noextract
fn t_read_1
  (#a:Type u#0) {| sized a |}
  (#sz : erased nat)
  (arr : array1 a (l1_forward sz))
  (i : SZ.t { SZ.v i < sz })
  (dummy : a)
  (#f : perm)
  (#va : chest1 a sz)
  preserves cpu ** on gpu_loc (arr |-> Frac f va)
  returns x : a
  ensures pure (x == acc1 va i)
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
  lem_blit_index_0 (Seq.create 1 dummy) (to_seq (l1_forward sz) va) i;
  lem_index_chest1 va i;
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
let ce_term_fold (c : pos) (#n : nat) (rp : Seq.lseq real n) (r : nat) (t : nat)
  : Lemma (requires t < c)
          (ensures ce_term_r c rp r t ==
                   (0.0R -. acc1 (LSM.log_softmax_real
                              (seq_to_chest1 (crow rp c r))) t))
  = ()

(* The device-to-device row copy lands exactly row [r] (= [crow]). *)
let memcpy_row_eq
  (#n:nat) (c:nat) (prev : Seq.lseq f32 c) (s : Seq.lseq f32 n)
  (r:nat) (off:nat)
  : Lemma (requires off == r * c /\ r * c + c <= n)
          (ensures KS.seq_blit prev 0 s off c ==
                   Seq.slice s (r * c) (r * c + c))
  = Seq.lemma_eq_intro (KS.seq_blit prev 0 s off c)
      (Seq.slice s (r * c) (r * c + c))

(* Total accessor for the (host) per-row loss vector. *)
noextract
let sidx (s : Seq.seq f32) (r : nat) : f32 =
  if r < Seq.length s then Seq.index s r else zero

(* Total accessor for the target index of row [r] (as a nat). *)
noextract
let tgt (#b : nat) (stv : Seq.lseq SZ.t b) (r : nat) : nat =
  if r < b then SZ.v (stv @! r) else 0

(* For every already-processed row [r < bound], the stored per-row loss
   approximates the genuine real CE term of row [r] at its target. *)
let ce_carried
  (b : nat) (c : pos) (#n : nat)
  (rp : Seq.lseq real n) (stv : Seq.lseq SZ.t b)
  (vt : Seq.seq f32) (bound : nat)
  : prop =
  forall (r : nat). r < bound ==>
    sidx vt r %~ ce_term_r c rp r (tgt stv r)

(* Loop-entry witness (vacuous at [bound = 0]). *)
let ce_inv_init
  (b : nat) (c : pos) (#n : nat)
  (rp : Seq.lseq real n) (stv : Seq.lseq SZ.t b)
  (vt : Seq.seq f32)
  : Lemma (ce_carried b c rp stv vt 0)
  = ()

(* Per-iteration extension: storing the row-[vi] loss into slot [vi]
   extends the agreement prefix from [< vi] to [< vi+1]. *)
let ce_prefix_extend
  (b : nat) (c : pos) (#n : nat)
  (rp : Seq.lseq real n) (stv : Seq.lseq SZ.t b)
  (vt vt' : Seq.seq f32) (vi : nat { vi < b }) (newv : f32)
  : Lemma
    (requires
      Seq.length vt == b /\ Seq.length vt' == b /\
      vt' == Seq.upd vt vi newv /\
      newv %~ ce_term_r c rp vi (tgt stv vi) /\
      ce_carried b c rp stv vt vi)
    (ensures ce_carried b c rp stv vt' (vi + 1))
  = ()

let ce_carried_complete
  (b : pos) (c : pos)
  (rp : Seq.lseq real (b * c)) (stv : Seq.lseq SZ.t b)
  (vt : Seq.lseq f32 b)
  : Lemma (requires ce_carried b c rp stv vt b)
          (ensures vt %~ real_cross_entropy_terms b c rp stv)
  = introduce forall (r:nat{r < b}).
      (vt @! r) %~ (real_cross_entropy_terms b c rp stv @! r)
    with ()

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn ce_loss_impl
  (b : szp { b <= max_blocks * max_threads /\
             SZ.fits (b + max_threads) })
  (c : szp { c <= max_blocks * max_threads /\
             SZ.fits (c + max_threads) /\
             SZ.fits (b * c) })
  (predictions : array1 f32 (l1_forward (b * c)) { is_global predictions })
  (targets : array1 SZ.t (l1_forward b) { is_global targets })
  (#sp : chest1 f32 (b * c))
  (#stv : chest1 SZ.t b)
  (rp : erased (Seq.lseq real (b * c)))
  (#fp #ft : perm)
  norewrite
  preserves cpu **
            on gpu_loc (predictions |-> Frac fp sp) **
            on gpu_loc (targets |-> Frac ft stv) **
            pure (sp %~ seq_to_chest1 rp)
  requires
    pure (forall (r : nat). r < SZ.v b ==> SZ.v (acc1 (reveal stv) r) < SZ.v c)
  returns res : f32
  ensures
    pure (cross_entropy_post b c rp
            (chest1_to_seq (reveal stv) <: Seq.lseq SZ.t b)
            res)
{
  let stv_s : erased (Seq.lseq SZ.t b) =
    hide (chest1_to_seq (reveal stv));
  let inv_b = ce_recip_f32 b;
  let terms : erased (Seq.lseq real b) =
    hide (real_cross_entropy_terms b c rp (reveal stv_s));

  let scratch = alloc0 #f32 c (l1_forward c);
  let t_dev   = alloc0 #f32 b (l1_forward b);
  let t_host  = Vec.alloc #f32 (zero #f32) b;

  ce_inv_init b c rp (reveal stv_s)
    (Seq.create b (zero #f32));
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
              ce_carried b c rp (reveal stv_s) vt vi)
    decreases (SZ.v b - SZ.v !idx)
  {
    let i = !idx;
    let off : SZ.t = SZ.(i *^ c);
    assert pure (SZ.v off == SZ.v i * SZ.v c);
    FStar.Math.Lemmas.lemma_mult_le_right c (SZ.v i + 1) b;
    assert pure (SZ.v i * SZ.v c + SZ.v c <= SZ.v b * SZ.v c);
    assert pure (i * c + c <= b * c);

    (* ── copy row [i] into scratch ──────────────────────────────── *)
    with va_prev. assert (on gpu_loc (scratch |-> reveal va_prev));
    t_memcpy_d2d' scratch 0sz predictions off c;
    with sca. assert (on gpu_loc (scratch |-> reveal sca));
    assert pure (chest1_to_seq (reveal sca) ==
                 KS.seq_blit (chest1_to_seq (reveal va_prev)) 0
                             (chest1_to_seq (reveal sp)) off c);
    memcpy_row_eq #(b * c) c
      (chest1_to_seq (reveal va_prev)) (chest1_to_seq (reveal sp)) i off;
    assert pure (chest1_to_seq (reveal sca) ==
                 Seq.slice (chest1_to_seq (reveal sp)) (i * c) (i * c + c));
    chest_to_seq_approx (reveal sp) rp;
    row_slice_approx (chest1_to_seq (reveal sp)) rp c i;
    chest_from_seq_approx (reveal sca)
      (Seq.slice (chest1_to_seq (reveal sp)) (i * c) (i * c + c))
      (crow rp c i);

    (* ── verified numerically-stable log-softmax in place ───────── *)
    let ra : chest1 real c = hide (seq_to_chest1 (crow rp c i));
    assert pure (reveal sca %~ reveal ra);
    LSM.log_softmax_gpu #f32 1024sz scratch ra;
    with sca'. assert (on gpu_loc (scratch |-> reveal sca'));
    assert pure (reveal sca' %~ LSM.log_softmax_real (reveal ra));

    (* ── gather the (negated) target lane ───────────────────────── *)
    let ti = t_read_1 targets i 0sz;
    lem_index_chest1 (reveal stv) i;
    assert pure (ti == acc1 (reveal stv) i);
    assert pure (SZ.v ti < SZ.v c);
    let v = t_read_1 scratch ti (zero #f32);
    assert pure (v == acc1 (reveal sca') ti);
    (* pointwise: scratch'[ti] approximates log_softmax(...)[ti] *)
    approx_at (reveal sca') (LSM.log_softmax_real (reveal ra)) ti;
    assert pure (v %~ acc1 (LSM.log_softmax_real (reveal ra)) ti);
    let neg_v : f32 = sub (zero #f32) v;
    neg_approx_f32 v (acc1 (LSM.log_softmax_real (reveal ra)) ti);
    (* fold back to the spec term [ce_term_r] *)
    ce_term_fold c rp i ti;
    assert pure (neg_v %~ ce_term_r c rp i ti);
    assert pure (tgt (reveal stv_s) i == SZ.v ti);
    assert pure (neg_v %~ ce_term_r c rp i (tgt (reveal stv_s) i));

    (* ── store per-row loss ─────────────────────────────────────── *)
    Vec.pts_to_len t_host;
    Vec.(t_host.(i) <- neg_v);
    with vt_old. assert (Vec.pts_to t_host (reveal vt_old));
    ce_prefix_extend b c rp (reveal stv_s)
      (reveal vt_old) (Seq.upd (reveal vt_old) i neg_v)
      i neg_v;

    idx := SZ.(!idx +^ 1sz);
  };

  (* ── on-device reduce-sum of the B per-row losses ───────────────── *)
  with vt_loop. assert (Vec.pts_to t_host (reveal vt_loop));
  Vec.pts_to_len t_host;
  t_memcpy_h2d t_dev t_host b;
  with vt_dev_final. assert (on gpu_loc (t_dev |-> reveal vt_dev_final));
  assert pure (chest1_to_seq (reveal vt_dev_final) == reveal vt_loop);

  ce_carried_complete b c rp (reveal stv_s)
    (reveal vt_loop <: Seq.lseq f32 b);
  chest_from_seq_approx (reveal vt_dev_final)
    (reveal vt_loop <: Seq.lseq f32 b) (reveal terms);
  let vr : chest1 real b = hide (seq_to_chest1 (reveal terms));
  let s = HRed.reduce #f32 id id 1024sz b t_dev vr;
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  assert pure (Seq.equal (chest1_to_seq (reveal vr)) (reveal terms));
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
  real_cross_entropy_mul b c rp (reveal stv_s);
  assert pure (m %~ real_cross_entropy b c rp (reveal stv_s));

  Vec.free t_host;
  free scratch;
  free t_dev;
  m;
}
#pop-options

#push-options "--z3rlimit 40"
let ce_loss_fw_f32 : ce_loss_fw_ty =
  fun b c predictions targets #sp #stv rp #fp #ft ->
    ce_loss_impl b c predictions targets #sp #stv rp #fp #ft
#pop-options
