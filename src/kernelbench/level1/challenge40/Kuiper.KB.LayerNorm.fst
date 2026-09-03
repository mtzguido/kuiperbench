module Kuiper.KB.LayerNorm

#lang-pulse
open Kuiper
open Kuiper.Float.Casts
open Kuiper.Scalars.Ops
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Approximates.Base
open Kuiper.Spec.Frobenius
open Kuiper.Spec.LayerNorm
module SZ = Kuiper.SizeT
module Copy = Kuiper.KB.Tensor.Copy
module F32 = Kuiper.Float32
module Map = Kuiper.Kernel.Map
module KBMap = Kuiper.KB.Compat.Map
module HRed = Kuiper.Kernel.HReduce
module KS = Kuiper.Seq.Common
module RsqrtApprox = Kuiper.KB.Compat.RsqrtApprox

(* Proof-local description of the concrete floating intermediates.  Its final
   conjunct records the direct-real public row contract, so none of these
   numerical witnesses escape through the entry point. *)
let row_layer_normalized
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#bn:nat) (#n:nat)
  (sx sx':Seq.lseq t bn) (gamma beta:Seq.lseq t n)
  (off:nat{off+n <= bn}) (eps inv_n:t) : prop =
  exists (sum sumsq mean m2 var var_eps inv neg_mean_inv:t).
    (let row = Seq.slice sx off (off+n) in
     sum %~ rsum (to_real_seq row) /\
     sumsq %~ frobenius_sumsq_r (to_real_seq row) /\
     mean == mul sum inv_n /\
     m2 == mul sumsq inv_n /\
     var == sub m2 (mul mean mean) /\
     var_eps == add var eps /\
     inv == rsqrt var_eps /\
     neg_mean_inv == sub (zero #t) (mul mean inv) /\
     Seq.slice sx' off (off+n) ==
       ln_row_result #t #_ #n inv neg_mean_inv gamma beta row /\
     (n > 0 ==>
       Kuiper.Spec.LayerNorm.row_layer_normalized
         sx sx' gamma beta off eps))

let layernorm_float_post
  (b n:nat) (eps inv_n:f32)
  (gamma beta:Seq.lseq f32 n)
  (sx sx':Seq.lseq f32 (b*n)) : prop =
  forall (r:nat). r < b ==>
    r*n+n <= b*n /\
    row_layer_normalized sx sx' gamma beta (r*n) eps inv_n

let row_layer_real_from_witnesses
  (#bn:nat) (n:pos)
  (gamma beta:Seq.lseq f32 n)
  (sx sx':Seq.lseq f32 bn) (off:nat{off+n <= bn})
  (eps inv_n sum sumsq mean m2 var var_eps inv neg_mean_inv:f32)
  : Lemma
      (requires
        sum %~ rsum (to_real_seq (Seq.slice sx off (off+n))) /\
        sumsq %~ frobenius_sumsq_r
          (to_real_seq (Seq.slice sx off (off+n))) /\
        inv_n %~ (1.0R /. FStar.Real.of_int n) /\
        mean == mul sum inv_n /\ m2 == mul sumsq inv_n /\
        var == sub m2 (mul mean mean) /\ var_eps == add var eps /\
        inv == rsqrt var_eps /\
        neg_mean_inv == sub (zero #f32) (mul mean inv) /\
        Seq.slice sx' off (off+n) ==
          ln_row_result #f32 #_ #n inv neg_mean_inv gamma beta
            (Seq.slice sx off (off+n)) /\
        row_layernorm_domain sx off n eps)
      (ensures Kuiper.Spec.LayerNorm.row_layer_normalized
        sx sx' gamma beta off eps)
  = let row = to_real_seq (Seq.slice sx off (off+n)) in
    let rmean = ln_mean_r #n row in
    let rm2 = ln_m2_r #n row in
    let rarg = ln_arg_r #n (to_real eps) row in
    a_mul sum inv_n (rsum row) (1.0R /. FStar.Real.of_int n);
    assert (mean %~ rmean);
    a_mul sumsq inv_n (frobenius_sumsq_r row)
      (1.0R /. FStar.Real.of_int n);
    assert (m2 %~ rm2);
    a_mul mean mean rmean rmean;
    sub_approx m2 (mul mean mean) rm2 (rmean *. rmean);
    to_real_ok eps;
    a_add var eps (rm2 -. rmean *. rmean) (to_real eps);
    assert (var_eps %~ rarg);
    RsqrtApprox.rsqrt_approx var_eps rarg;
    let rinv : real = FStar.Math.Sqrt.rsqrt rarg in
    assert (inv %~ rinv);
    a_mul mean inv rmean rinv;
    sub_approx (zero #f32) (mul mean inv) 0.0R (rmean *. rinv);
    assert (neg_mean_inv %~ (0.0R -. rmean *. rinv));
    let aux (j:nat{j<n}) : Lemma
      (Seq.index (Seq.slice sx' off (off+n)) j %~
       Seq.index
         (ln_row_result_r #n (to_real eps)
           (to_real_seq gamma) (to_real_seq beta) row) j)
      = let x = Seq.index (Seq.slice sx off (off+n)) j in
        let g = Seq.index gamma j in
        let b = Seq.index beta j in
        let rx = Seq.index row j in
        let rg = Seq.index (to_real_seq gamma) j in
        let rb = Seq.index (to_real_seq beta) j in
        to_real_ok x;
        to_real_ok g;
        to_real_ok b;
        a_mul x inv rx rinv;
        a_add (mul x inv) neg_mean_inv (rx *. rinv)
          (0.0R -. rmean *. rinv);
        a_mul (add (mul x inv) neg_mean_inv) g
          (rx *. rinv +. (0.0R -. rmean *. rinv)) rg;
        a_add (mul (add (mul x inv) neg_mean_inv) g) b
          ((rx *. rinv +. (0.0R -. rmean *. rinv)) *. rg) rb
    in
    Classical.forall_intro aux

(* ── l1_forward <-> seq bridges (row memcpy / reduce / map) ────────────

   The GPU buffers here use the identity row-major layout [l1_forward],
   whose index map is the identity, so its [to_seq]/[from_seq] flattening
   coincides with [chest1_to_seq]/[seq_to_chest1].  These pointwise
   identities are discharged by extensionality. *)

(* [to_seq] on [l1_forward] equals [chest1_to_seq]. *)
let lem_to_seq (#et:Type) (n:nat) (c : chest1 et n)
  : Lemma (to_seq (l1_forward n) c == chest1_to_seq c)
  = assert (Seq.equal (to_seq (l1_forward n) c) (chest1_to_seq c))

(* [chest1_to_seq] commutes with [chest_map]/[seq_map]. *)
let chest_map_to_seq (#et1 #et2 : Type) (#n : nat)
  (f : et1 -> et2) (c : chest1 et1 n)
  : Lemma (chest1_to_seq (chest_map f c) == KS.seq_map f (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (chest_map f c)) (KS.seq_map f (chest1_to_seq c)))

(* [chest1_to_seq] commutes with [to_real_chest]/[to_real_seq]. *)
let to_real_chest_to_seq (#et : Type0) {| scalar et, real_like et |} (#n : nat)
  (c : chest1 et n)
  : Lemma (chest1_to_seq (to_real_chest c) == to_real_seq (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (to_real_chest c)) (to_real_seq (chest1_to_seq c)))

(* [chest1_to_seq] commutes with the two-argument [chest1_map2]/[lseq_map2].
   Bridges the [map_gpu2] chest-level result back to the seq-level
   [KBMap.lseq_map2] used by [ln_row_result_via_affine_lemma]. *)
let chest1_map2_to_seq (#et : Type) (#n : nat)
  (f : et -> et -> et) (a b : chest1 et n)
  : Lemma (chest1_to_seq (Map.chest1_map2 f a b) ==
           KBMap.lseq_map2 f (chest1_to_seq a) (chest1_to_seq b))
  = assert (Seq.equal (chest1_to_seq (Map.chest1_map2 f a b))
                      (KBMap.lseq_map2 f (chest1_to_seq a) (chest1_to_seq b)))

(* Device-to-device offset blit at the chest level.  Bridges the raw
   [Kuiper.KB.Compat.Array.gpu_memcpy_device_to_device'] primitive (which works on the backing
   [core] larrays) through [tensor_concr]/[tensor_abs]/[tensor_abs'] and
   the [l1_forward] round-trip.  This is the direct replacement for the
   deleted [Array1.memcpy_device_to_device'] and keeps the SAME argument
   order (dst, dst_off, src, src_off, cnt).  [src] is preserved; [dst] is
   overwritten by the blit at offset [dst_off]. *)
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

let seq_map_id_eq (#a:Type) (s : Seq.seq a)
  : Lemma (Seq.equal (KS.seq_map id s) s)
  = ()

(* Slice of a self-blit recovers the blitted content.  Proved in a clean
   context (away from the heavy [layer_norm_row] proof) so that the slice
   equality is available robustly via a single lemma instantiation rather
   than re-derived under hundreds of in-scope hypotheses. *)
#push-options "--z3rlimit 30 --fuel 1 --ifuel 1"
let blit_self_slice (#a:Type) (s1 : Seq.seq a) (off : nat) (content : Seq.seq a)
  : Lemma (requires off + Seq.length content <= Seq.length s1)
          (ensures
             Seq.slice (KS.seq_blit s1 off content 0 (Seq.length content))
               off (off + Seq.length content) == content)
  =
  let n = Seq.length content in
  Seq.lemma_eq_intro (Seq.slice content 0 n) content;
  Seq.lemma_eq_intro
    (Seq.slice (KS.seq_blit s1 off content 0 n) off (off + n))
    content
#pop-options

(* Pointwise square approximation lemma. *)
let sq_step_approx
  (#t:Type0) {| scalar t, real_like t |}
  (x : t) (r : real)
  : Lemma (requires v_approximates x r)
          (ensures  v_approximates (square x) (sq_step_r r))
  = a_mul x x r r

let sq_step_approx_forall (#t:Type0) {| scalar t, real_like t |} ()
  : Lemma (square #t %~ sq_step_r)
  = Classical.forall_intro_2
      (fun (xv:t) ->
         Classical.move_requires (sq_step_approx #t xv))

(* Shape of the per-row result: applying [affine_result inv b],
   then mul-broadcast gamma, then add-broadcast beta, equals [ln_row_result]. *)
let ln_row_result_via_affine_lemma
  (n : nat)
  (inv neg_mean_inv : f32)
  (gamma beta row : Seq.lseq f32 n)
  : Lemma
      (KBMap.lseq_map2 add
         (KBMap.lseq_map2 mul
            (affine_result #f32 inv neg_mean_inv #n row)
            gamma)
         beta
       ==
       ln_row_result #_ #_ #n inv neg_mean_inv gamma beta row)
  = let lhs : Seq.lseq f32 n =
      KBMap.lseq_map2 add
        (KBMap.lseq_map2 mul
           (affine_result #f32 inv neg_mean_inv #n row)
           gamma)
        beta in
    let rhs : Seq.lseq f32 n =
      ln_row_result #_ #_ #n inv neg_mean_inv gamma beta row in
    let aux (j : nat { j < n })
      : Lemma (Seq.index lhs j == Seq.index rhs j)
      = () in
    Classical.forall_intro aux;
    Seq.lemma_eq_intro lhs rhs

(* Intro lemma for row_layer_normalized: witnesses → predicate. *)
#push-options "--z3rlimit 20"
let row_layer_normalized_intro
  (#bn : nat) (n : pos)
  (gamma beta : Seq.lseq f32 n)
  (sx sx' : Seq.lseq f32 bn)
  (off : nat { off + n <= bn })
  (eps inv_n : f32)
  (sum sumsq mean m2 var var_eps inv neg_mean_inv : f32)
  : Lemma
      (requires
        sum   %~ rsum (to_real_seq (Seq.slice sx off (off + n))) /\
        sumsq %~ frobenius_sumsq_r (to_real_seq (Seq.slice sx off (off + n))) /\
        mean         == mul sum   inv_n /\
        m2           == mul sumsq inv_n /\
        var          == sub m2 (mul mean mean) /\
        var_eps      == add var eps /\
        inv          == rsqrt var_eps /\
        neg_mean_inv == sub (zero #f32) (mul mean inv) /\
        Seq.slice sx' off (off + n) ==
          ln_row_result #_ #_ #n inv neg_mean_inv gamma beta
            (Seq.slice sx off (off + n)) /\
        inv_n %~ (1.0R /. FStar.Real.of_int n) /\
        row_layernorm_domain sx off n eps)
      (ensures row_layer_normalized sx sx' gamma beta off eps inv_n)
  = row_layer_real_from_witnesses n gamma beta sx sx' off eps inv_n
      sum sumsq mean m2 var var_eps inv neg_mean_inv
#pop-options

(* Per-row body: copy x[r,:] into scratch, sum-reduce for the mean,
   sum-reduce of squared values for the second moment, apply affine
   (inv, -mean*inv) in place, broadcast-mul by gamma, broadcast-add
   beta, then copy scratch back into x[r,:]. *)
#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn layer_norm_row
  (b : szp)
  (n : szp { n <= max_blocks * max_threads /\
             b * n <= max_blocks * max_threads })
  (rv_off : sz { rv_off + n <= b * n })
  (eps : f32)
  (inv_n : f32)
  (x : array1 f32 (l1_forward (b * n)) { is_global x })
  (gamma : array1 f32 (l1_forward n) { is_global gamma })
  (beta  : array1 f32 (l1_forward n) { is_global beta  })
  (scratch : array1 f32 (l1_forward n) { is_global scratch })
  (#fg #fb : perm)
  (#sx : chest1 f32 (b * n))
  (#sg : chest1 f32 n)
  (#sbeta : chest1 f32 n)
  (#ss : chest1 f32 n)
  preserves cpu
  preserves
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sbeta)
  requires
    on gpu_loc (x |-> sx) ** on gpu_loc (scratch |-> ss) **
    pure (inv_n %~ (1.0R /. FStar.Real.of_int (SZ.v n))) **
    pure (row_layernorm_domain (chest1_to_seq sx) rv_off n eps)
  ensures
    (exists* (sx' : chest1 f32 (b * n)) (ss' : chest1 f32 n).
       on gpu_loc (x |-> sx') ** on gpu_loc (scratch |-> ss') **
       pure (row_layer_normalized #_ #_ #_ #_ #_ #n
               (chest1_to_seq sx) (chest1_to_seq sx')
               (chest1_to_seq sg) (chest1_to_seq sbeta) rv_off eps inv_n /\
             (exists (inv neg_mean_inv : f32).
                chest1_to_seq sx' == KS.seq_blit (chest1_to_seq sx) rv_off
                         (ln_row_result #_ #_ #n inv neg_mean_inv
                            (chest1_to_seq sg) (chest1_to_seq sbeta)
                            (Seq.slice (chest1_to_seq sx) rv_off (SZ.v rv_off + SZ.v n)))
                         0 n)))
{
  (* Pass 1: copy row x[rv_off .. rv_off+n) into scratch[0 .. n). *)
  t_memcpy_d2d' scratch 0sz x rv_off n;
  with vs1. assert (on gpu_loc (scratch |-> reveal vs1));
  (* [seq_blit ss 0 sx rv_off n] fully overwrites the length-n scratch,
     so it equals [slice sx rv_off (rv_off+n)] -- the row. *)
  let row_g : erased (lseq f32 n) =
    hide (Seq.slice (chest1_to_seq (reveal sx)) rv_off (SZ.v rv_off + SZ.v n));
  Seq.lemma_eq_intro
    (KS.seq_blit (chest1_to_seq (reveal ss)) 0 (chest1_to_seq (reveal sx)) rv_off n)
    (reveal row_g);
  assert pure (chest1_to_seq (reveal vs1) == reveal row_g);

  (* Real-valued view of the (preserved) scratch row, shared by both
     reductions.  [reduce] preserves [scratch |-> vs1]. *)
  let vr : chest1 real n = hide (to_real_chest (reveal vs1));
  assert pure (reveal vs1 %~ reveal vr);

  (* Pass 1 reduce: plain sum over the row -> sum1. *)
  let sum1 = HRed.reduce #f32 id id 1024sz n scratch #vs1 vr;
  chest_map_to_seq id (reveal vr);
  seq_map_id_eq #real (chest1_to_seq (reveal vr) <: Seq.seq real);
  to_real_chest_to_seq (reveal vs1);
  assert pure (sum1 %~ rsum (to_real_seq (reveal row_g)));
  let mean = mul sum1 inv_n;

  (* Pass 2: sum reduce of squared values via [pre_map = square]. *)
  sq_step_approx_forall #f32 ();
  let sum2 = HRed.reduce #f32 (square #f32) sq_step_r 1024sz n scratch #vs1 vr;
  chest_map_to_seq sq_step_r (reveal vr);
  assert pure (sum2 %~ frobenius_sumsq_r (to_real_seq (reveal row_g)));
  let m2 = mul sum2 inv_n;
  let var = sub m2 (mul mean mean);
  let var_eps = add var eps;
  let inv = rsqrt var_eps;
  let neg_mean_inv = sub (zero #f32) (mul mean inv);

  (* Pass 3: scratch still contains vs1; apply per-row affine. *)
  Map.map_gpu (affine_step inv neg_mean_inv) n scratch;

  (* Pass 4: scratch *= gamma. *)
  Map.map_gpu2 #f32 mul n scratch gamma;

  (* Pass 5: scratch += beta. *)
  Map.map_gpu2 #f32 add n scratch beta;
  with sfin. assert (on gpu_loc (scratch |-> reveal sfin));

  (* Bridge the chest-level scratch content back to the seq-level golden
     spec.  The composed maps flatten (via [chest_map]/[chest1_map2] ->
     [seq_map]/[lseq_map2]) to
       lseq_map2 add (lseq_map2 mul (affine_result inv nmi row_g) gamma) beta,
     which [ln_row_result_via_affine_lemma] rewrites to [ln_row_result]. *)
  chest_map_to_seq (affine_step inv neg_mean_inv) (reveal vs1);
  chest1_map2_to_seq mul (chest_map (affine_step inv neg_mean_inv) (reveal vs1)) (reveal sg);
  chest1_map2_to_seq add
    (Map.chest1_map2 mul (chest_map (affine_step inv neg_mean_inv) (reveal vs1)) (reveal sg))
    (reveal sbeta);
  ln_row_result_via_affine_lemma n inv neg_mean_inv
    (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta)) (reveal row_g);
  assert pure (chest1_to_seq (reveal sfin) ==
               ln_row_result #_ #_ #n inv neg_mean_inv
                 (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta)) (reveal row_g));

  (* Pass 6: copy scratch back into x[rv_off .. rv_off+n). *)
  t_memcpy_d2d' x rv_off scratch 0sz n;
  with vfinal. assert (on gpu_loc (x |-> reveal vfinal));

  (* Prove the concrete value of vfinal at the seq level. *)
  Seq.lemma_eq_intro
    (chest1_to_seq (reveal vfinal))
    (KS.seq_blit (chest1_to_seq (reveal sx)) rv_off
       (ln_row_result #_ #_ #n inv neg_mean_inv
          (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta)) (reveal row_g))
       0 n);

  (* Prove the slice equality needed for row_layer_normalized_intro.
     [vfinal] equals the self-blit of [ln_row_result] into [sx] at
     [rv_off] (proven just above), so slicing [vfinal] back out at
     [rv_off, rv_off+n] recovers [ln_row_result].  Use the standalone
     [blit_self_slice] lemma so this holds robustly. *)
  assert pure (Seq.length (chest1_to_seq (reveal vfinal)) == b * n);
  assert pure (rv_off + n <= b * n);
  blit_self_slice (chest1_to_seq (reveal sx)) rv_off
    (ln_row_result #_ #_ #n inv neg_mean_inv
       (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta)) (reveal row_g));
  Seq.lemma_eq_intro
    (Seq.slice (chest1_to_seq (reveal vfinal)) rv_off (SZ.v rv_off + SZ.v n))
    (ln_row_result #_ #_ #n inv neg_mean_inv
       (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta)) (reveal row_g));

  (* Introduce row_layer_normalized. *)
  (* Tie [reveal row_g] to the slice the lemma's [requires] is stated over, so
     the [%~] facts and the slice equality line up with the lemma conjuncts. *)
  assert pure (rv_off + n <= b * n);
  assert pure (reveal row_g ==
               Seq.slice (chest1_to_seq (reveal sx)) rv_off (SZ.v rv_off + SZ.v n));
  assert pure (sum1 %~ rsum (to_real_seq (reveal row_g)));
  assert pure (sum2 %~ frobenius_sumsq_r (to_real_seq (reveal row_g)));
  row_layer_normalized_intro #(b * n) n
    (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta))
    (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal vfinal))
    rv_off eps inv_n
    sum1 sum2 mean m2 var var_eps inv neg_mean_inv;

  (* Witness the blit existential. *)
  assert pure (exists (i nmi : f32).
                 chest1_to_seq (reveal vfinal) ==
                   KS.seq_blit (chest1_to_seq (reveal sx)) rv_off
                     (ln_row_result #_ #_ #n i nmi
                        (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta))
                        (Seq.slice (chest1_to_seq (reveal sx)) rv_off (SZ.v rv_off + SZ.v n)))
                     0 n);
  ()
}
#pop-options

(* ===== Generic blit-slice helpers (same as MeanVarNorm) ===== *)

let blit_slice_left
  (#a:Type) (s1 s2 : Seq.seq a) (off1 off2 cnt lo hi : nat)
  : Lemma (requires off1 + cnt <= Seq.length s1 /\
                    off2 + cnt <= Seq.length s2 /\
                    lo <= hi /\ hi <= off1)
          (ensures
            Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) lo hi ==
            Seq.slice s1 lo hi)
  = Seq.lemma_eq_intro
      (Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) lo hi)
      (Seq.slice s1 lo hi)

let blit_slice_right
  (#a:Type) (s1 s2 : Seq.seq a) (off1 off2 cnt lo hi : nat)
  : Lemma (requires off1 + cnt <= Seq.length s1 /\
                    off2 + cnt <= Seq.length s2 /\
                    off1 + cnt <= lo /\ lo <= hi /\ hi <= Seq.length s1)
          (ensures
            Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) lo hi ==
            Seq.slice s1 lo hi)
  = Seq.lemma_eq_intro
      (Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) lo hi)
      (Seq.slice s1 lo hi)

let blit_slice_inside
  (#a:Type) (s1 s2 : Seq.seq a) (off1 off2 cnt : nat)
  : Lemma (requires off1 + cnt <= Seq.length s1 /\
                    off2 + cnt <= Seq.length s2)
          (ensures
            Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) off1 (off1 + cnt) ==
            Seq.slice s2 off2 (off2 + cnt))
  = Seq.lemma_eq_intro
      (Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) off1 (off1 + cnt))
      (Seq.slice s2 off2 (off2 + cnt))

(* ===== Blit step lemma for LayerNorm ===== *)

(* Given that sx_pre's suffix matches sx from vi*n onward, and sx_post
   is the blit result, derive suffix/prefix slice preservation. *)
let ln_blit_step_lemma
  (b n : nat)
  (gamma beta : Seq.lseq f32 n)
  (sx sx_pre sx_post : Seq.lseq f32 (b * n))
  (vi : nat)
  (inv neg_mean_inv : f32)
  : Lemma
    (requires
      vi < b /\
      Seq.slice sx_pre (vi * n) (b * n) ==
        Seq.slice sx (vi * n) (b * n) /\
      sx_post ==
        KS.seq_blit sx_pre (vi * n)
          (ln_row_result #_ #_ #n inv neg_mean_inv gamma beta
             (Seq.slice sx_pre (vi * n) (vi * n + n))) 0 n)
    (ensures
      Seq.slice sx_post ((vi + 1) * n) (b * n) ==
        Seq.slice sx ((vi + 1) * n) (b * n) /\
      Seq.slice sx_post (vi * n) (vi * n + n) ==
        ln_row_result #_ #_ #n inv neg_mean_inv gamma beta
          (Seq.slice sx (vi * n) (vi * n + n)) /\
      (forall (r : nat). r * n + n <= vi * n ==>
        Seq.slice sx_post (r * n) (r * n + n) ==
        Seq.slice sx_pre  (r * n) (r * n + n)))
  = let row_pre  : Seq.lseq f32 n =
      Seq.slice sx_pre (vi * n) (vi * n + n) in
    let row_orig : Seq.lseq f32 n =
      Seq.slice sx     (vi * n) (vi * n + n) in
    FStar.Seq.Properties.slice_slice sx_pre (vi * n) (b * n) 0 n;
    FStar.Seq.Properties.slice_slice sx     (vi * n) (b * n) 0 n;
    Seq.lemma_eq_intro row_pre row_orig;
    let result_seq : Seq.seq f32 =
      ln_row_result #_ #_ #n inv neg_mean_inv gamma beta row_pre in
    blit_slice_inside #f32 sx_pre result_seq (vi * n) 0 n;
    Seq.lemma_eq_intro
      (Seq.slice sx_post (vi * n) (vi * n + n))
      result_seq;
    blit_slice_right #f32 sx_pre result_seq
      (vi * n) 0 n ((vi + 1) * n) (b * n);
    let len_tail = b * n - vi * n in
    FStar.Seq.Properties.slice_slice sx_pre (vi * n) (b * n) n len_tail;
    FStar.Seq.Properties.slice_slice sx     (vi * n) (b * n) n len_tail;
    Seq.lemma_eq_intro
      (Seq.slice sx_pre ((vi + 1) * n) (b * n))
      (Seq.slice sx ((vi + 1) * n) (b * n));
    let bn : nat = b * n in
    FStar.Math.Lemmas.lemma_mult_le_right n vi b;
    let vin : nat = vi * n in
    let sx_post' : Seq.lseq f32 bn = sx_post in
    let sx_pre'  : Seq.lseq f32 bn = sx_pre in
    let aux (r : nat{r * n + n <= bn}) : Lemma
      (requires r * n + n <= vin)
      (ensures Seq.slice sx_post' (r * n) (r * n + n) ==
               Seq.slice sx_pre'  (r * n) (r * n + n))
      = blit_slice_left #f32 sx_pre' result_seq
          vin 0 n (r * n) (r * n + n)
    in
    let aux2 (r : nat) : Lemma
      (requires r * n + n <= vin)
      (ensures r * n + n <= bn /\
               Seq.slice sx_post' (r * n) (r * n + n) ==
               Seq.slice sx_pre'  (r * n) (r * n + n))
      = aux r
    in
    Classical.forall_intro (Classical.move_requires aux2)

(* Peel the [exists inv neg_mean_inv] from the row's postcondition and
   apply ln_blit_step_lemma to obtain prefix/suffix preservation. *)
let ln_loop_step_lemma
  (b n : nat)
  (gamma beta : Seq.lseq f32 n)
  (sx sx_pre sx_post : Seq.lseq f32 (b * n))
  (vi : nat)
  : Lemma
    (requires
      vi < b /\
      Seq.slice sx_pre (vi * n) (b * n) ==
        Seq.slice sx (vi * n) (b * n) /\
      (exists (inv neg_mean_inv : f32).
         sx_post == KS.seq_blit sx_pre (vi * n)
           (ln_row_result #_ #_ #n inv neg_mean_inv gamma beta
             (Seq.slice sx_pre (vi * n) (vi * n + n))) 0 n))
    (ensures
      vi * n + n <= b * n /\
      Seq.slice sx_post ((vi + 1) * n) (b * n) ==
        Seq.slice sx ((vi + 1) * n) (b * n) /\
      (forall (r : nat). r * n + n <= vi * n ==>
        Seq.slice sx_post (r * n) (r * n + n) ==
        Seq.slice sx_pre  (r * n) (r * n + n)))
  = FStar.Math.Lemmas.lemma_mult_le_right n (vi + 1) b;
    FStar.Seq.Properties.slice_slice sx_pre (vi * n) (b * n) 0 n;
    FStar.Seq.Properties.slice_slice sx     (vi * n) (b * n) 0 n;
    Seq.lemma_eq_intro
      (Seq.slice sx_pre (vi * n) (vi * n + n))
      (Seq.slice sx     (vi * n) (vi * n + n));
    let p (inv neg_mean_inv : f32) : prop =
      sx_post == KS.seq_blit sx_pre (vi * n)
        (ln_row_result #_ #_ #n inv neg_mean_inv gamma beta
          (Seq.slice sx_pre (vi * n) (vi * n + n))) 0 n
    in
    let goal : prop =
      vi * n + n <= b * n /\
      Seq.slice sx_post ((vi + 1) * n) (b * n) ==
        Seq.slice sx ((vi + 1) * n) (b * n) /\
      (forall (r : nat). r * n + n <= vi * n ==>
        Seq.slice sx_post (r * n) (r * n + n) ==
        Seq.slice sx_pre  (r * n) (r * n + n))
    in
    let inner (inv neg_mean_inv : f32) : Lemma
      (requires p inv neg_mean_inv)
      (ensures goal)
      = ln_blit_step_lemma b n gamma beta sx sx_pre sx_post vi inv neg_mean_inv
    in
    Classical.forall_to_exists #f32
      #(fun (inv : f32) -> exists (nmi : f32). p inv nmi)
      #goal
      (fun (inv : f32) ->
         Classical.forall_to_exists #f32 #(p inv) #goal
           (fun (nmi : f32) ->
              Classical.move_requires (inner inv) nmi))

(* Single-row lift: if sx and sx_alt agree on row r, and the row is
   normalized w.r.t. sx_alt, it's also normalized w.r.t. sx. *)
#push-options "--z3rlimit 30 --fuel 2 --ifuel 2"
let rln_lift_input
  (#bn : nat) (n : nat) (off : nat{off + n <= bn}) (eps inv_n : f32)
  (gamma beta : Seq.lseq f32 n)
  (sx sx_alt sx_post : Seq.lseq f32 bn)
  : Lemma
    (requires
      row_layer_normalized sx_alt sx_post gamma beta off eps inv_n /\
      Seq.slice sx off (off + n) == Seq.slice sx_alt off (off + n))
    (ensures row_layer_normalized sx sx_post gamma beta off eps inv_n)
  = ()
#pop-options

(* Extract a per-row slice equality from a suffix-slice equality, then
   apply rln_lift_input. *)
#push-options "--z3rlimit 30"
let rln_lift_input_via_suffix
  (#bn : nat) (n : nat) (vi : nat)
  (suffix_lo : nat{vi * n + n <= suffix_lo /\ suffix_lo <= bn})
  (eps inv_n : f32)
  (gamma beta : Seq.lseq f32 n)
  (sx sx_alt sx_post : Seq.lseq f32 bn)
  : Lemma
    (requires
      row_layer_normalized sx_alt sx_post gamma beta (vi * n) eps inv_n /\
      Seq.slice sx (vi * n) suffix_lo ==
        Seq.slice sx_alt (vi * n) suffix_lo)
    (ensures row_layer_normalized sx sx_post gamma beta (vi * n) eps inv_n)
  = FStar.Seq.Properties.slice_slice sx     (vi * n) suffix_lo 0 n;
    FStar.Seq.Properties.slice_slice sx_alt (vi * n) suffix_lo 0 n;
    rln_lift_input n (vi * n) eps inv_n gamma beta sx sx_alt sx_post
#pop-options

let row_layernorm_domain_via_suffix
  (#bn:nat) (n:pos) (vi:nat)
  (suffix_hi:nat{vi*n+n <= suffix_hi /\ suffix_hi <= bn})
  (eps:f32) (sx sx_alt:Seq.lseq f32 bn)
  : Lemma
      (requires
        row_layernorm_domain sx (vi*n) n eps /\
        Seq.slice sx (vi*n) suffix_hi ==
          Seq.slice sx_alt (vi*n) suffix_hi)
      (ensures row_layernorm_domain sx_alt (vi*n) n eps)
  = FStar.Seq.Properties.slice_slice sx (vi*n) suffix_hi 0 n;
    FStar.Seq.Properties.slice_slice sx_alt (vi*n) suffix_hi 0 n

(* For all r < vi, r*n+n <= vi*n. *)
let ln_prefix_le_lemma (n vi r : nat)
  : Lemma
    (requires r < vi)
    (ensures r * n + n <= vi * n)
  = FStar.Math.Lemmas.lemma_mult_le_right n (r + 1) vi

let ln_prefix_le_forall_lemma (n vi : nat)
  : Lemma (ensures forall (r : nat). r < vi ==> r * n + n <= vi * n)
  = let aux (r : nat) : Lemma
      (requires r < vi)
      (ensures r * n + n <= vi * n)
    = ln_prefix_le_lemma n vi r
    in
    Classical.forall_intro (Classical.move_requires aux)

(* For all r < b, r*n+n <= b*n. Proved once in a clean context. *)
let row_lt_b_bound_forall_lemma (n b : nat)
  : Lemma (ensures forall (r : nat). r < b ==> r * n + n <= b * n)
  = let aux (r : nat{r < b}) : Lemma (ensures r * n + n <= b * n) =
      if n = 0 then ()
      else FStar.Math.Lemmas.lemma_mult_le_right n (r + 1) b
    in
    Classical.forall_intro aux

(* Transfer row_layer_normalized from sx_pre to sx_post for all rows
   r < vi, given per-row slice equalities. *)
#push-options "--z3rlimit 60"
let transfer_rln_forall
    (bn n vi : nat)
    (gamma beta : Seq.lseq f32 n)
    (sx sx_pre sx_post : Seq.lseq f32 bn)
    (eps inv_n : f32)
  : Lemma
    (requires
      vi * n <= bn /\
      (forall (r : nat). r < vi ==> r * n + n <= bn) /\
      (forall (r : nat). r < vi ==>
         row_layer_normalized sx sx_pre gamma beta (r * n) eps inv_n) /\
      (forall (r : nat). r < vi ==>
         Seq.slice sx_post (r * n) (r * n + n) ==
         Seq.slice sx_pre  (r * n) (r * n + n)))
    (ensures
      forall (r : nat). r < vi ==>
        row_layer_normalized sx sx_post gamma beta (r * n) eps inv_n)
  = let aux (r : nat{r < vi}) : Lemma
        (ensures row_layer_normalized sx sx_post gamma beta (r * n) eps inv_n) =
      ()
    in
    Classical.forall_intro (Classical.move_requires aux)
#pop-options

(* Extend the per-row normalisation invariant from vi rows to vi+1. *)
#push-options "--z3rlimit 60"
let extend_row_ln_forall
    (bn n b vi : nat)
    (gamma beta : Seq.lseq f32 n)
    (sx sx' : Seq.lseq f32 bn)
    (eps inv_n : f32)
  : Lemma
    (requires
      vi < b /\
      0 < n /\
      bn = b * n /\
      vi * n + n <= bn /\
      (forall (r : nat). r < vi ==> r * n + n <= bn) /\
      (forall (r : nat). r < vi ==>
         row_layer_normalized sx sx' gamma beta (r * n) eps inv_n) /\
      row_layer_normalized sx sx' gamma beta (vi * n) eps inv_n)
    (ensures
      forall (r : nat). r < vi + 1 ==>
        r * n + n <= bn /\
        row_layer_normalized sx sx' gamma beta (r * n) eps inv_n)
  = let aux (r : nat{r < vi + 1}) : Lemma
        (ensures r * n + n <= bn /\
                 row_layer_normalized sx sx' gamma beta (r * n) eps inv_n) =
      if r < vi then ()
      else ()  (* r = vi *)
    in
    Classical.forall_intro aux
#pop-options

(* Whole-tensor entry point: loop over rows. *)
inline_for_extraction noextract
#push-options "--z3rlimit 60"
fn layer_norm
  (b : szp)
  (n : szp { n <= max_blocks * max_threads /\
             SZ.fits (b * n) /\
             b * n <= max_blocks * max_threads })
  (eps : f32)
  (inv_n : f32)
  (x : array1 f32 (l1_forward (b * n)) { is_global x })
  (gamma : array1 f32 (l1_forward n) { is_global gamma })
  (beta  : array1 f32 (l1_forward n) { is_global beta  })
  (#fg #fb : perm)
  (#sx : chest1 f32 (b * n))
  (#sg : chest1 f32 n)
  (#sbeta : chest1 f32 n)
  preserves
    cpu **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sbeta)
  requires
    on gpu_loc (x |-> sx) **
    pure (inv_n %~ (1.0R /. FStar.Real.of_int (SZ.v n))) **
    pure (layernorm_domain b n eps (chest1_to_seq sx))
  ensures
    (exists* (sx' : chest1 f32 (b * n)).
       on gpu_loc (x |-> sx') **
       pure (layernorm_float_post b n eps inv_n
               (chest1_to_seq sg) (chest1_to_seq sbeta)
               (chest1_to_seq sx) (chest1_to_seq sx')))
{
  let scratch = alloc0 #f32 n (l1_forward n);
  let mut idx = 0sz;
  (* Establish forall r < b. r*n+n <= b*n in a clean context BEFORE
     the loop, so the fact is available both inside the loop body and
     after the loop for layernorm_post. *)
  row_lt_b_bound_forall_lemma n b;
  while (let i = !idx; SZ.(i <^ b))
    invariant
      exists* (vi : sz) (sx' : chest1 f32 (b * n)) (ss' : chest1 f32 n).
        idx |-> vi **
        on gpu_loc (x |-> sx') **
        on gpu_loc (scratch |-> ss') **
        cpu **
        pure (SZ.v vi <= SZ.v b /\
              (forall (r : nat). r < SZ.v vi ==>
                 row_layer_normalized #_ #_ #_ #_ #_ #n
                   (chest1_to_seq sx) (chest1_to_seq sx')
                   (chest1_to_seq sg) (chest1_to_seq sbeta) (r * SZ.v n) eps inv_n) /\
              Seq.slice (chest1_to_seq sx') (SZ.v vi * SZ.v n) (SZ.v b * SZ.v n) ==
                Seq.slice (chest1_to_seq sx) (SZ.v vi * SZ.v n) (SZ.v b * SZ.v n))
    decreases (SZ.v b - SZ.v !idx)
  {
    let i = !idx;
    let off : sz = SZ.(i *^ n);
    with sx_pre. assert (on gpu_loc (x |-> reveal sx_pre));
    (* Capture the loop invariant's per-row normalisation forall as a
       stable pure fact BEFORE the [layer_norm_row] call.  Right after
       loop-body entry the chest congruence [chest1_to_seq sx' ==
       chest1_to_seq (reveal sx_pre)] fires cheaply; once the row call
       grows the proof context Z3 can no longer re-derive it, so we
       persist it here (it feeds transfer_rln_forall's 3rd precondition). *)
    assert pure (forall (r : nat). r < SZ.v i ==>
      row_layer_normalized #_ #_ #_ #_ #_ #n
        (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx_pre))
        (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta))
        (r * SZ.v n) eps inv_n);
    (* Re-establish the bound fact inside the loop body. *)
    row_lt_b_bound_forall_lemma n b;
    row_layernorm_domain_via_suffix #(b*n) n i (b*n) eps
      (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx_pre));
    layer_norm_row b n off eps inv_n x gamma beta scratch;
    with sx_post. assert (on gpu_loc (x |-> reveal sx_post));
    (* From the row postcondition: row is normalized w.r.t. sx_pre,
       and sx_post is the blit.  Use ln_loop_step_lemma to get
       suffix slice preservation and prefix slice preservation. *)
    ln_loop_step_lemma b n (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta))
      (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx_pre)) (chest1_to_seq (reveal sx_post)) i;
    (* Lift the just-completed row's predicate from sx_pre to sx. *)
    rln_lift_input_via_suffix n i (SZ.v b * SZ.v n)
      eps inv_n (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta))
      (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx_pre)) (chest1_to_seq (reveal sx_post));
    assert pure (row_layer_normalized #_ #_ #_ #_ #_ #n
                   (chest1_to_seq sx) (chest1_to_seq (reveal sx_post))
                   (chest1_to_seq sg) (chest1_to_seq sbeta) (SZ.v i * SZ.v n) eps inv_n);
    (* Establish r*n+n <= vi*n for all r < vi. *)
    ln_prefix_le_forall_lemma n i;
    (* Stage transfer_rln_forall's 4th precondition (per-row slice
       preservation) in a clean sub-query.  Its 3rd precondition (the
       per-row normalisation forall) was already captured pre-row-call
       above and persists here. *)
    assert pure (forall (r : nat). r < SZ.v i ==>
      r * n + n <= b * n /\
      Seq.slice (chest1_to_seq (reveal sx_post)) (r * SZ.v n) (r * SZ.v n + SZ.v n) ==
      Seq.slice (chest1_to_seq (reveal sx_pre)) (r * SZ.v n) (r * SZ.v n + SZ.v n));
    (* Transfer row_layer_normalized from sx_pre to sx_post for all
       rows r < i.  Uses a pure F* lemma in a clean context. *)
    transfer_rln_forall (b * n) n i
      (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta))
      (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx_pre)) (chest1_to_seq (reveal sx_post)) eps inv_n;
    (* Extend the invariant from i rows to i+1. *)
    extend_row_ln_forall (b * n) n b i
      (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta))
      (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx_post)) eps inv_n;
    assert pure (SZ.v i + 1 == SZ.v (SZ.add i 1sz));
    idx := SZ.(!idx +^ 1sz);
  };
  free scratch;
  ()
}
#pop-options

let ln_inv_n_approx (n:szp)
  : Lemma (ln_inv_n #f32 n %~
      (1.0R /. FStar.Real.of_int (SZ.v n)))
  = let n64 : Int64.t = FStar.Int.Cast.uint64_to_int64
      (FStar.SizeT.sizet_to_uint64 n) in
    assert (Int64.v n64 == SZ.v n);
    of_int_approx #f32 n64;
    div_approx (one #f32) (of_int #f32 n64)
      1.0R (FStar.Real.of_int (SZ.v n))

let layernorm_float_post_to_real
  (b:nat) (n:pos) (eps inv_n:f32)
  (gamma beta:Seq.lseq f32 n)
  (sx sx':Seq.lseq f32 (b*n))
  : Lemma
      (requires layernorm_float_post b n eps inv_n gamma beta sx sx')
      (ensures layernorm_post b n eps gamma beta sx sx')
  = ()

(* Public entry point: compute the per-row reciprocal [ln_inv_n n] inside
   the verification boundary (extracts to 1.0f / (float)(int64_t)(uint64_t)n),
   then delegate to [layer_norm].  The proof above treats [inv_n]
   abstractly, so this constant computation does not affect its cost. *)
fn layernorm_fw
  (b : szp)
  (n : szp { n <= max_blocks * max_threads /\
             SZ.fits (b * n) /\
             b * n <= max_blocks * max_threads })
  (eps : f32)
  (x : array1 f32 (l1_forward (b * n)) { is_global x })
  (gamma : array1 f32 (l1_forward n) { is_global gamma })
  (beta  : array1 f32 (l1_forward n) { is_global beta  })
  (#fg #fb : perm)
  (#sx : chest1 f32 (b * n))
  (#sg : chest1 f32 n)
  (#sbeta : chest1 f32 n)
  preserves
    cpu **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sbeta)
  requires
    on gpu_loc (x |-> sx) **
    pure (layernorm_domain b n eps (chest1_to_seq sx))
  ensures
    (exists* (sx' : chest1 f32 (b * n)).
       on gpu_loc (x |-> sx') **
       pure (layernorm_post b n eps
               (chest1_to_seq sg) (chest1_to_seq sbeta)
               (chest1_to_seq sx) (chest1_to_seq sx')))
{
  let inv_n : f32 = ln_inv_n n;
  ln_inv_n_approx n;
  layer_norm b n eps inv_n x gamma beta;
  with sx'. assert (on gpu_loc (x |-> reveal sx'));
  layernorm_float_post_to_real b n eps inv_n
    (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sbeta))
    (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx'));
}

let layernorm_fw_f32 = layernorm_fw

fn layernorm4d_alloc_f32
  (b c h w : szp)
  (eps : f64)
  (x : array1 f32 (l1_forward (b * (c * (h * w)))) { is_global x })
  (gamma : array1 f32 (l1_forward (c * (h * w))) { is_global gamma })
  (beta : array1 f32 (l1_forward (c * (h * w))) { is_global beta })
  (#fx #fg #fb : perm)
  (#sx : chest1 f32 (b * (c * (h * w))))
  (#sg #sb : chest1 f32 (c * (h * w)))
  preserves
    cpu ** on gpu_loc (x |-> Frac fx sx) **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta |-> Frac fb sb)
  requires
    pure (SZ.fits (h * w) /\ SZ.fits (c * (h * w)) /\
          c * (h * w) <= max_blocks * max_threads /\
          SZ.fits (b * (c * (h * w))) /\
          b * (c * (h * w)) <= max_blocks * max_threads /\
          layernorm_domain b (c * (h * w)) (fcast #f64 #f32 eps)
            (chest1_to_seq sx))
  returns out : array1 f32 (l1_forward (b * (c * (h * w))))
  ensures
    exists* (sx' : chest1 f32 (b * (c * (h * w)))).
      on gpu_loc (out |-> sx') **
      pure (layernorm_post b (c * (h * w)) (fcast #f64 #f32 eps)
              (chest1_to_seq sg) (chest1_to_seq sb)
              (chest1_to_seq sx) (chest1_to_seq sx'))
{
  let eps32 : f32 = fcast eps;
  let hw : szp = h *^ w;
  let n : szp = c *^ hw;
  let elems : szp = b *^ n;
  let out = Copy.copy_alloc #f32 elems x;
  layernorm_fw_f32 b n eps32 out gamma beta;
  out
}
