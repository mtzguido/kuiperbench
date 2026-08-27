module Kuiper.KB.L2Norm

#lang-pulse
open Kuiper
open Kuiper.Scalars.Ops
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Approximates.Base
open Kuiper.Spec.Frobenius
open Kuiper.Spec.L2Norm
module SZ = Kuiper.SizeT
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module KS = Kuiper.Seq.Common

(* Local lemmas about slice-of-blit used by the host loop. *)
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

(* Plain (no refinement) lemma capturing what the post of l2norm_row gives
   us about how a single iteration relates sx_pre and sx_post.  Refines no
   spec-level row predicate; just produces the slice equalities the SMT
   solver needs to maintain the host-loop invariant. *)
let l2_blit_step_lemma
  (b d : nat)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (vi : nat)
  (inv sumsq : f32)
  : Lemma
    (requires
      vi < b /\
      Seq.slice sx_pre (vi * d) (b * d) ==
        Seq.slice sx (vi * d) (b * d) /\
      sx_post ==
        KS.seq_blit sx_pre (vi * d)
          (frobenius_result #f32 inv #d
             (Seq.slice sx_pre (vi * d) (vi * d + d))) 0 d)
    (ensures
      (* trailing slice unchanged: *)
      Seq.slice sx_post ((vi + 1) * d) (b * d) ==
        Seq.slice sx ((vi + 1) * d) (b * d) /\
      (* row vi equals frobenius_result inv (slice sx vi*d (vi+1)*d): *)
      Seq.slice sx_post (vi * d) (vi * d + d) ==
        frobenius_result #f32 inv #d
          (Seq.slice sx (vi * d) (vi * d + d)) /\
      (* prefix preserved: *)
      (forall (r : nat). r * d + d <= vi * d ==>
        Seq.slice sx_post (r * d) (r * d + d) ==
        Seq.slice sx_pre  (r * d) (r * d + d)))
  =
    let row_pre  : Seq.lseq f32 d =
      Seq.slice sx_pre (vi * d) (vi * d + d) in
    let row_orig : Seq.lseq f32 d =
      Seq.slice sx     (vi * d) (vi * d + d) in
    FStar.Seq.Properties.slice_slice sx_pre (vi * d) (b * d) 0 d;
    FStar.Seq.Properties.slice_slice sx     (vi * d) (b * d) 0 d;
    Seq.lemma_eq_intro row_pre row_orig;
    let result_seq : Seq.seq f32 =
      frobenius_result #f32 inv #d row_pre in
    blit_slice_inside #f32 sx_pre result_seq (vi * d) 0 d;
    Seq.lemma_eq_intro
      (Seq.slice sx_post (vi * d) (vi * d + d))
      result_seq;
    (* trailing slice via blit_slice_right + slice_slice on sx_pre/sx *)
    blit_slice_right #f32 sx_pre result_seq
      (vi * d) 0 d ((vi + 1) * d) (b * d);
    let len_tail = b * d - vi * d in
    FStar.Seq.Properties.slice_slice sx_pre (vi * d) (b * d) d len_tail;
    FStar.Seq.Properties.slice_slice sx     (vi * d) (b * d) d len_tail;
    Seq.lemma_eq_intro
      (Seq.slice sx_pre ((vi + 1) * d) (b * d))
      (Seq.slice sx ((vi + 1) * d) (b * d));
    (* prefix slices via blit_slice_left, parameterised over r *)
    let bd : nat = b * d in
    FStar.Math.Lemmas.lemma_mult_le_right d vi b;
    let vid : nat = vi * d in
    assert (vid <= bd);
    let sx_post' : Seq.lseq f32 bd = sx_post in
    let sx_pre'  : Seq.lseq f32 bd = sx_pre in
    let aux (r : nat{r * d + d <= bd}) : Lemma
      (requires r * d + d <= vid)
      (ensures Seq.slice sx_post' (r * d) (r * d + d) ==
               Seq.slice sx_pre'  (r * d) (r * d + d))
      = blit_slice_left #f32 sx_pre' result_seq
          vid 0 d (r * d) (r * d + d)
    in
    let aux2 (r : nat) : Lemma
      (requires r * d + d <= vid)
      (ensures r * d + d <= bd /\
               Seq.slice sx_post' (r * d) (r * d + d) ==
               Seq.slice sx_pre'  (r * d) (r * d + d))
      = aux r
    in
    Classical.forall_intro (Classical.move_requires aux2)

(* Prove (r+1)*d <= vi*d follows from r < vi for d : nat. *)
let prefix_le_lemma (d vi r : nat)
  : Lemma
    (requires r < vi)
    (ensures r * d + d <= vi * d)
  = FStar.Math.Lemmas.lemma_mult_le_right d (r + 1) vi

(* Per-row lift: if a row was already L2-normalised in sx_pre and the slice
   at that row didn't change, it remains L2-normalised in sx_post. *)
let row_l2_normalized_lift
  (b d : nat)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (vi : nat)
  (r : nat)
  : Lemma
    (requires
      r < vi /\ vi <= b /\
      row_l2_normalized sx sx_pre (r * d) d /\
      Seq.slice sx_post (r * d) (r * d + d) ==
        Seq.slice sx_pre  (r * d) (r * d + d))
    (ensures
      r * d + d <= b * d /\
      row_l2_normalized sx sx_post (r * d) d)
  = FStar.Math.Lemmas.lemma_mult_le_right d (r + 1) vi;
    FStar.Math.Lemmas.lemma_mult_le_right d vi b

(* Convenience wrapper that internally elims the (inv, sumsq) existential
   and produces [row_l2_normalized sx sx_post (vi*d) d]. *)
(* [--z3refresh] + [] keep each sub-VC small and give
   Z3 a fresh solver state: the combined query here tripped a Z3 4.13.3
   internal arithmetic crash (lar_solver: m_columns_with_changed_bounds). *)
#push-options "--z3rlimit 40 --z3refresh"
let l2_loop_step_lemma
  (b d : nat)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (vi : nat)
  : Lemma
    (requires
      vi < b /\
      Seq.slice sx_pre (vi * d) (b * d) ==
        Seq.slice sx (vi * d) (b * d) /\
      (exists (inv : f32) (sumsq : f32).
         sumsq %~ frobenius_sumsq_r
                    (to_real_seq
                      (Seq.slice sx_pre (vi * d) (vi * d + d))) /\
         inv == rsqrt sumsq /\
         sx_post ==
           KS.seq_blit sx_pre (vi * d)
             (frobenius_result #f32 inv #d
                (Seq.slice sx_pre (vi * d) (vi * d + d))) 0 d))
    (ensures
      vi * d + d <= b * d /\
      row_l2_normalized sx sx_post (vi * d) d /\
      Seq.slice sx_post ((vi + 1) * d) (b * d) ==
        Seq.slice sx ((vi + 1) * d) (b * d) /\
      (forall (r : nat). r * d + d <= vi * d ==>
        Seq.slice sx_post (r * d) (r * d + d) ==
        Seq.slice sx_pre  (r * d) (r * d + d)))
  =
    FStar.Math.Lemmas.lemma_mult_le_right d (vi + 1) b;
    FStar.Seq.Properties.slice_slice sx_pre (vi * d) (b * d) 0 d;
    FStar.Seq.Properties.slice_slice sx     (vi * d) (b * d) 0 d;
    Seq.lemma_eq_intro
      (Seq.slice sx_pre (vi * d) (vi * d + d))
      (Seq.slice sx     (vi * d) (vi * d + d));
    let aux (inv sumsq : f32)
      : Lemma
        ((sumsq %~ frobenius_sumsq_r
                     (to_real_seq
                       (Seq.slice sx_pre (vi * d) (vi * d + d))) /\
          inv == rsqrt sumsq /\
          sx_post ==
            KS.seq_blit sx_pre (vi * d)
              (frobenius_result #f32 inv #d
                 (Seq.slice sx_pre (vi * d) (vi * d + d))) 0 d)
         ==>
         (Seq.slice sx_post ((vi + 1) * d) (b * d) ==
            Seq.slice sx ((vi + 1) * d) (b * d) /\
          sumsq %~ frobenius_sumsq_r
                     (to_real_seq
                       (Seq.slice sx (vi * d) (vi * d + d))) /\
          inv == rsqrt sumsq /\
          Seq.slice sx_post (vi * d) (vi * d + d) ==
            frobenius_result #f32 inv #d
              (Seq.slice sx (vi * d) (vi * d + d)) /\
          (forall (r : nat). r * d + d <= vi * d ==>
            Seq.slice sx_post (r * d) (r * d + d) ==
            Seq.slice sx_pre  (r * d) (r * d + d))))
      = Classical.move_requires
          (l2_blit_step_lemma b d sx sx_pre sx_post vi inv) sumsq
    in
    Classical.forall_intro_2 aux
#pop-options

(* Top-level per-r helper for the new invariant.  Hoisted out of the body
   of l2_loop_invariant_step so SMT sees a clean, narrow context. *)

(* Case r == vi: directly inherit row_l2_normalized from the precondition. *)
#push-options "--z3rlimit 30"
let row_normalized_at_vi
  (b d vi : nat)
  (sx sx_post : Seq.lseq f32 (b * d))
  : Lemma
    (requires
      vi < b /\
      vi * d <= b * d /\
      row_l2_normalized sx sx_post (vi * d) d)
    (ensures
      vi * d + d <= b * d /\
      row_l2_normalized sx sx_post (vi * d) d)
  = FStar.Math.Lemmas.lemma_mult_le_right d (vi + 1) b

(* Case r < vi: lift via the prefix-preservation hypothesis. *)
let row_normalized_at_r_lt_vi
  (b d vi : nat)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (r : nat)
  : Lemma
    (requires
      r < vi /\ vi <= b /\
      row_l2_normalized sx sx_pre (r * d) d /\
      (forall (r' : nat). r' * d + d <= vi * d ==>
         r' * d + d <= b * d /\
         Seq.slice sx_post (r' * d) (r' * d + d) ==
         Seq.slice sx_pre  (r' * d) (r' * d + d)))
    (ensures
      r * d + d <= b * d /\
      row_l2_normalized sx sx_post (r * d) d)
  = prefix_le_lemma d vi r;
    row_l2_normalized_lift b d sx sx_pre sx_post vi r
#pop-options

#push-options "--z3rlimit 30 --z3refresh"
(* Squeeze two nat bounds into an equality in a clean, noise-free context.
   Under the heavy nonlinear context of row_normalized_at_r_new (vi*d, b*d,
   ...), SMT can establish r >= vi and r <= vi individually but fails to
   combine them into r == vi; proving it here in isolation is robust. *)
let nat_squeeze (r vi : nat)
  : Lemma (requires ~(r < vi) /\ r < vi + 1) (ensures r == vi)
  = ()

let row_normalized_at_r_new
  (b d vi : nat)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (r : nat)
  : Lemma
    (requires
      vi < b /\ r < vi + 1 /\
      vi * d <= b * d /\
      row_l2_normalized sx sx_post (vi * d) d /\
      (forall (r' : nat). r' < vi ==>
         row_l2_normalized sx sx_pre (r' * d) d) /\
      (forall (r' : nat). r' * d + d <= vi * d ==>
         r' * d + d <= b * d /\
         Seq.slice sx_post (r' * d) (r' * d + d) ==
         Seq.slice sx_pre  (r' * d) (r' * d + d)))
    (ensures
      r * d + d <= b * d /\
      row_l2_normalized sx sx_post (r * d) d)
  =
    FStar.Math.Lemmas.lemma_mult_le_right d (vi + 1) b;
    FStar.Math.Lemmas.lemma_mult_le_right d (r + 1) (vi + 1);
    if r < vi
    then row_normalized_at_r_lt_vi b d vi sx sx_pre sx_post r
    else begin
      nat_squeeze r vi;
      row_normalized_at_vi b d vi sx sx_post
    end

#pop-options

(* Single combined lemma that re-establishes the host loop invariant after
   one iteration.  Putting everything into one lemma avoids the fragile
   forall-composition of three independent Classical.forall_intro calls. *)
#push-options "--z3rlimit 100"
let l2_loop_invariant_step
  (b d : nat)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (vi : nat)
  : Lemma
    (requires
      vi < b /\
      (forall (r : nat). r < vi ==> row_l2_normalized sx sx_pre (r * d) d) /\
      Seq.slice sx_pre (vi * d) (b * d) ==
        Seq.slice sx (vi * d) (b * d) /\
      (exists (inv : f32) (sumsq : f32).
         sumsq %~ frobenius_sumsq_r
                    (to_real_seq
                      (Seq.slice sx_pre (vi * d) (vi * d + d))) /\
         inv == rsqrt sumsq /\
         sx_post ==
           KS.seq_blit sx_pre (vi * d)
             (frobenius_result #f32 inv #d
                (Seq.slice sx_pre (vi * d) (vi * d + d))) 0 d))
    (ensures
      vi + 1 <= b /\
      (forall (r : nat). r < vi + 1 ==>
         row_l2_normalized sx sx_post (r * d) d) /\
      Seq.slice sx_post ((vi + 1) * d) (b * d) ==
        Seq.slice sx ((vi + 1) * d) (b * d))
  =
    l2_loop_step_lemma b d sx sx_pre sx_post vi;
    FStar.Math.Lemmas.lemma_mult_le_right d vi b;
    FStar.Math.Lemmas.lemma_mult_le_right d (vi + 1) b;
    (* Forall hint: r * d + d <= b * d for every r <= vi. *)
    let mul_bound (r : nat) : Lemma (requires r <= vi) (ensures r * d + d <= b * d) =
      FStar.Math.Lemmas.lemma_mult_le_right d (r + 1) b
    in
    Classical.forall_intro (Classical.move_requires mul_bound);
    Classical.forall_intro
      (Classical.move_requires
        (row_normalized_at_r_new b d vi sx sx_pre sx_post))
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

inline_for_extraction noextract
fn l2norm_row
  (b : szp)
  (d : szp { d <= max_blocks * max_threads /\
             b * d <= max_blocks * max_threads })
  (rv_off : sz { rv_off + d <= b * d })
  (x : array1 f32 (l1_forward (b * d)) { is_global x })
  (scratch : array1 f32 (l1_forward d) { is_global scratch })
  (#sx : chest1 f32 (b * d))
  (#ss : chest1 f32 d)
  preserves cpu
  requires on gpu_loc (x |-> sx) ** on gpu_loc (scratch |-> ss)
  ensures
    (exists* (sx' : chest1 f32 (b * d)) (ss' : chest1 f32 d).
       on gpu_loc (x |-> sx') ** on gpu_loc (scratch |-> ss') **
       pure (exists (inv : f32) (sumsq : f32).
         (let row : Seq.lseq f32 d =
            Seq.slice (chest1_to_seq sx) rv_off (SZ.v rv_off + SZ.v d) in
          sumsq %~ frobenius_sumsq_r (to_real_seq row) /\
          inv == rsqrt sumsq /\
          chest1_to_seq sx' ==
            KS.seq_blit (chest1_to_seq sx) rv_off
              (frobenius_result #f32 inv #d row) 0 d)))
{
  (* Copy the row x[rv_off .. rv_off+d) into scratch[0 .. d). *)
  t_memcpy_d2d' scratch 0sz x rv_off d;
  with ss1. assert (on gpu_loc (scratch |-> reveal ss1));
  (* [seq_blit ss 0 sx rv_off d] fully overwrites the length-d scratch,
     so it equals [slice sx rv_off (rv_off+d)] -- the row. *)
  let row_g : erased (lseq f32 d) =
    hide (Seq.slice (chest1_to_seq sx) rv_off (SZ.v rv_off + SZ.v d));
  Seq.lemma_eq_intro
    (KS.seq_blit (chest1_to_seq ss) 0 (chest1_to_seq sx) rv_off d)
    (reveal row_g);
  assert pure (chest1_to_seq (reveal ss1) == reveal row_g);

  (* Sum of squares over the row (device tree-reduce with a square pre-map). *)
  sq_step_approx_forall #f32 ();
  let vr : chest1 real d = to_real_chest (reveal ss1);
  assert pure (reveal ss1 %~ vr);
  let sumsq = HRed.reduce #f32 square sq_step_r 1024sz d scratch #ss1 vr;
  let inv = rsqrt sumsq;

  (* In-place scale of the (preserved) row by inv. *)
  Map.map_gpu (smul_step inv) d scratch;

  (* Copy the scaled row back into x[rv_off .. rv_off+d). *)
  t_memcpy_d2d' x rv_off scratch 0sz d;
  with sx'. assert (on gpu_loc (x |-> reveal sx'));

  (* Bridge the chest-level results back to the seq-level golden spec:
     scratch = chest_map (smul_step inv) ss1 flattens to
     frobenius_result inv row, and the reduce's sum-of-squares over
     [to_real_chest ss1] flattens to frobenius_sumsq_r (to_real_seq row). *)
  chest_map_to_seq (smul_step inv) (reveal ss1);
  chest_map_to_seq sq_step_r vr;
  to_real_chest_to_seq (reveal ss1);
  assert pure (chest1_to_seq vr == to_real_seq (reveal row_g));
  assert pure (sumsq %~ frobenius_sumsq_r (to_real_seq (reveal row_g)));
  assert pure (
    chest1_to_seq (chest_map (smul_step inv) (reveal ss1)) ==
    frobenius_result #f32 inv #d (reveal row_g));
  FStar.Classical.exists_intro
    (fun (sumsq' : f32) ->
      sumsq' %~ frobenius_sumsq_r (to_real_seq (reveal row_g)) /\
      inv == rsqrt sumsq' /\
      chest1_to_seq (reveal sx') ==
        KS.seq_blit (chest1_to_seq sx) rv_off
          (frobenius_result #f32 inv #d (reveal row_g)) 0 d)
    sumsq;
  FStar.Classical.exists_intro
    (fun (inv' : f32) -> exists (sumsq' : f32).
      sumsq' %~ frobenius_sumsq_r (to_real_seq (reveal row_g)) /\
      inv' == rsqrt sumsq' /\
      chest1_to_seq (reveal sx') ==
        KS.seq_blit (chest1_to_seq sx) rv_off
          (frobenius_result #f32 inv' #d (reveal row_g)) 0 d)
    inv;
  FStar.Seq.lemma_len_slice (chest1_to_seq sx)
    rv_off (SZ.v rv_off + SZ.v d);
  assert pure (SZ.v rv_off + SZ.v d - SZ.v rv_off == SZ.v d);
  assert pure (Seq.length (Seq.slice (chest1_to_seq sx)
    rv_off (SZ.v rv_off + SZ.v d)) == SZ.v d);
  ()
}

#push-options "--z3rlimit 50 --z3refresh --z3seed 2"
inline_for_extraction noextract
fn l2norm
  (b : szp)
  (d : szp { d <= max_blocks * max_threads /\
             SZ.fits (b * d) /\
             b * d <= max_blocks * max_threads })
  (x : array1 f32 (l1_forward (b * d)) { is_global x })
  (#sx : chest1 f32 (b * d))
  preserves cpu
  requires on gpu_loc (x |-> sx)
  ensures
    (exists* (sx' : chest1 f32 (b * d)).
       on gpu_loc (x |-> sx') **
       pure (l2norm_post b d (chest1_to_seq sx) (chest1_to_seq sx')))
{
  let scratch = alloc0 #f32 d (l1_forward d);
  let mut idx = 0sz;
  while (let i = !idx; SZ.(i <^ b))
    invariant
      exists* (vi : sz) (sx' : chest1 f32 (b * d)) (ss' : chest1 f32 d).
        idx |-> vi **
        on gpu_loc (x |-> sx') **
        on gpu_loc (scratch |-> ss') **
        cpu **
        pure (SZ.v vi <= SZ.v b /\
              (forall (r : nat). r < SZ.v vi ==>
                 row_l2_normalized (chest1_to_seq sx) (chest1_to_seq sx') (r * SZ.v d) d) /\
              Seq.slice (chest1_to_seq sx') (SZ.v vi * SZ.v d) (SZ.v b * SZ.v d) ==
                Seq.slice (chest1_to_seq sx) (SZ.v vi * SZ.v d) (SZ.v b * SZ.v d))
    decreases (SZ.v b - SZ.v !idx)
  {
    let i = !idx;
    let off : sz = SZ.(i *^ d);
    with sx_pre. assert (on gpu_loc (x |-> reveal sx_pre));
    l2norm_row b d off x scratch;
    with sx_post. assert (on gpu_loc (x |-> reveal sx_post));
    l2_loop_invariant_step b d (chest1_to_seq (reveal sx))
      (chest1_to_seq (reveal sx_pre)) (chest1_to_seq (reveal sx_post)) i;
    idx := SZ.(!idx +^ 1sz);
  };
  free scratch;
  ()
}
#pop-options

let l2norm_fw_f32 : l2norm_fw_ty f32 =
  fun b d x #s -> l2norm b d x #s
