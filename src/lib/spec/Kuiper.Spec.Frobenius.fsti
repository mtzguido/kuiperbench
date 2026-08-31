module Kuiper.Spec.Frobenius

(* Functional specification for Frobenius-norm normalisation
   (KernelBench L1 #37):

       y_i = x_i / sqrt(sum_j x_j^2)

   The public contract directly relates the output to the real-valued
   mathematical normalization of the input.  The floating tree-reduction
   result and reciprocal square root remain proof-local; neither is exposed
   as an existential witness.

   The real definition is totalized with a zero branch, while verified
   callers require a positive squared norm.  This is necessary because the
   IEEE result of normalizing an all-zero input contains infinities/NaNs,
   which the finite-real approximation relation cannot describe. *)

open Kuiper.Scalars
open Kuiper.Real
open Kuiper.Approximates
module Seq = FStar.Seq
module KS = Kuiper.Seq.Common
module RealSqrt = FStar.Math.Sqrt

(* Per-element scaling step used by the kernel's final pass. *)
inline_for_extraction
let smul_step (#t:Type0) {| scalar t |} (c x : t) : t = mul x c

(* Real-arithmetic squaring step (the functional behaviour that the
   kernel's f32 squaring step approximates). *)
let sq_step_r (r : real) : real = r *. r

(* Real-arithmetic sum of squares of a real sequence -- the
   mathematical Frobenius squared-norm. *)
let frobenius_sumsq_r (s : Seq.seq real) : GTot real =
  rsum (KS.seq_map sq_step_r s)

val frobenius_sumsq_nonnegative
  (s : Seq.seq real)
  : Lemma (frobenius_sumsq_r s >=. 0.0R)

(* Direct real-valued Frobenius normalization.  The zero-norm branch makes
   the definition total; verified callers use the positive-norm branch,
   because IEEE [rsqrt 0] has no finite real interpretation. *)
let frobenius_inv_r (s : Seq.seq real) : real =
  let ss = frobenius_sumsq_r s in
  if ss >. 0.0R then RealSqrt.rsqrt ss else 0.0R

let frobenius_result_r
  (#n : nat)
  (s : Seq.lseq real n) : GTot (Seq.lseq real n) =
  KS.lseq_map (fun x -> x *. frobenius_inv_r s) s

let frobenius_post
  (#t:Type0) {| scalar t, real_like t |}
  (#n : nat)
  (s s' : Seq.lseq t n) : prop =
  (s' <: Seq.seq t) %~
    (frobenius_result_r #n (to_real_seq s) <: Seq.seq real)

(* Result of Frobenius normalisation given a *precomputed* reciprocal
   norm [inv].  Pointwise: [(frobenius_result inv s)_i = s_i * inv]. *)
let frobenius_result
  (#t:Type0) {| scalar t |}
  (inv : t)
  (#n : nat)
  (s : Seq.lseq t n) : GTot (Seq.lseq t n) =
  KS.lseq_map (smul_step inv) s

(* Convenience: result length is preserved.  This is immediate but
   convenient as a small lemma exposed via [.fsti]. *)
val frobenius_result_length
  (#t:Type0) {| scalar t |}
  (inv : t)
  (#n : nat)
  (s : Seq.lseq t n)
  : Lemma (Seq.length (frobenius_result inv s) == n)
          [SMTPat (Seq.length (frobenius_result inv s))]

(* Pointwise affine step: x -> a*x + b.  Shared by the row-normalisation
   kernels (#33 BatchNorm, #34/#35 mean/variance norm, #40 LayerNorm),
   where each element is rescaled by [a = 1/sigma], [b = -mu/sigma].
   Body is a single fused multiply-add; passed to [map_gpu]. *)
inline_for_extraction
let affine_step (#t:Type0) {| scalar t |} (a b : t) (x : t) : t =
  add (mul x a) b

(* Apply the affine step pointwise across a sequence.  This is exactly
   the postcondition shape of [map_gpu (affine_step a b)]. *)
let affine_result
  (#t:Type0) {| scalar t |}
  (a b : t)
  (#lena:nat)
  (s : Seq.lseq t lena) : GTot (Seq.lseq t lena) =
  KS.lseq_map (affine_step a b) s
