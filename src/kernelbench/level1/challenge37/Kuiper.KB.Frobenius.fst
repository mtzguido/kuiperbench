module Kuiper.KB.Frobenius

#lang-pulse
open Kuiper
open Kuiper.Scalars.Ops
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Approximates.Base
open Kuiper.Spec.Frobenius
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module KS = Kuiper.Seq.Common
module RsqrtApprox = Kuiper.KB.Compat.RsqrtApprox
module Copy = Kuiper.KB.Tensor.Copy

(* Pointwise approximation: [square x %~ sq_step_r r] whenever
   [x %~ r].  Direct consequence of [a_mul]. *)
let sq_step_approx
  (#t:Type0) {| scalar t, real_like t |}
  (x : t) (r : real)
  : Lemma (requires v_approximates x r)
          (ensures  v_approximates (square x) (sq_step_r r))
  = a_mul x x r r

(* Quantified version, instantiated lazily inside the kernel body
   so the SMT solver can discharge the [square %~ sq_step_r]
   precondition of [HRed.reduce] without manual bookkeeping. *)
let sq_step_approx_forall (#t:Type0) {| scalar t, real_like t |} ()
  : Lemma (square #t %~ sq_step_r)
  = Classical.forall_intro_2
      (fun (xv:t) ->
         Classical.move_requires (sq_step_approx #t xv))

(* [chest1_to_seq] commutes with [chest_map]/[seq_map] and with
   [to_real_chest]/[to_real_seq].  Both are pointwise identities over
   [Seq.init_ghost], discharged by extensionality. *)
let chest_map_to_seq (#et1 #et2 : Type) (#n : nat)
  (f : et1 -> et2) (c : chest1 et1 n)
  : Lemma (chest1_to_seq (chest_map f c) == Kuiper.Seq.Common.seq_map f (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (chest_map f c))
                      (Kuiper.Seq.Common.seq_map f (chest1_to_seq c)))

let to_real_chest_to_seq (#et : Type0) {| scalar et, real_like et |} (#n : nat)
  (c : chest1 et n)
  : Lemma (chest1_to_seq (to_real_chest c) == to_real_seq (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (to_real_chest c)) (to_real_seq (chest1_to_seq c)))

let frobenius_result_approx
  (#t:Type0) {| scalar t, real_like t |}
  (#n:nat)
  (inv : t) (rinv : real)
  (s : Seq.lseq t n) (rs : Seq.lseq real n)
  : Lemma
      (requires inv %~ rinv /\ s %~ rs)
      (ensures frobenius_result inv s %~
               KS.lseq_map (fun x -> x *. rinv) rs)
  = let lhs = frobenius_result inv s in
    let rhs = KS.lseq_map (fun x -> x *. rinv) rs in
    let aux (i:nat{i < n}) : Lemma ((lhs @! i) %~ (rhs @! i)) =
      a_mul (s @! i) inv (rs @! i) rinv
    in
    Classical.forall_intro aux

(* Frobenius normalisation, layout-fixed to [l1_forward] for
   extraction.  Composes [HRed.reduce] (with a square pre-map) and
   [Map.map_gpu] (with a scalar-multiply step). *)
inline_for_extraction noextract
fn frobenius
  (lena : szp { lena <= max_blocks * max_threads })
  (a : array1 f32 (l1_forward lena) { is_global a })
  (#va : chest1 f32 lena)
  preserves cpu
  requires
    on gpu_loc (a |-> va) **
    pure (frobenius_sumsq_r (to_real_seq (chest1_to_seq va)) >. 0.0R)
  ensures
    exists* (va' : chest1 f32 lena).
      on gpu_loc (a |-> va') **
      pure (frobenius_post (chest1_to_seq va) (chest1_to_seq va'))
{
  (* Establish the pointwise-square approximation fact in the
     ambient context so HRed.reduce's precondition discharges. *)
  sq_step_approx_forall #f32 ();

  (* Real spec of the input, used as the [vr] argument to reduce. *)
  let vr : chest1 real lena = hide (to_real_chest (reveal va));
  assert pure (reveal va %~ reveal vr);

  (* Sum of squares: pre-map = sq_step, no scratch buffer. *)
  let sumsq = HRed.reduce #f32 square sq_step_r 1024sz lena a #va vr;

  (* Reciprocal-norm scaling factor. *)
  let inv_norm = rsqrt sumsq;

  (* Single in-place pass: a[i] := a[i] * inv_norm. *)
  Map.map_gpu (smul_step inv_norm) lena a;

  (* Bridge the chest-level result and the sum-of-squares back to the
     seq-level golden spec. *)
  chest_map_to_seq (smul_step inv_norm) (reveal va);
  chest_map_to_seq sq_step_r (reveal vr);
  to_real_chest_to_seq (reveal va);
  let rss = frobenius_sumsq_r (to_real_seq (chest1_to_seq (reveal va)));
  assert pure (sumsq %~ rss);
  RsqrtApprox.rsqrt_approx sumsq rss;
  to_real_seq_is_approx (chest1_to_seq (reveal va));
  frobenius_result_approx inv_norm (FStar.Math.Sqrt.rsqrt rss)
    (chest1_to_seq (reveal va))
    (to_real_seq (chest1_to_seq (reveal va)));
  ()
}

let frobenius_fw_f32 : frobenius_fw_ty f32 = frobenius

fn frobenius_alloc_f32
  (lena : szp { lena <= max_blocks * max_threads })
  (a : array1 f32 (l1_forward lena) { is_global a })
  (#f : perm)
  (#s : chest1 f32 lena)
  preserves cpu ** on gpu_loc (a |-> Frac f s)
  requires
    pure (frobenius_sumsq_r (to_real_seq (chest1_to_seq s)) >. 0.0R)
  returns out : array1 f32 (l1_forward lena)
  ensures
    exists* (s' : chest1 f32 lena).
      on gpu_loc (out |-> s') **
      pure (frobenius_post (chest1_to_seq s) (chest1_to_seq s'))
{
  let out = Copy.copy_alloc #f32 lena a;
  frobenius_fw_f32 lena out;
  out
}
