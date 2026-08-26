module Kuiper.Spec.Frobenius

(* Functional specification for Frobenius-norm normalisation
   (KernelBench L1 #37):

       y_i = x_i / sqrt(sum_j x_j^2)

   At an *abstract*, [scalar]-polymorphic level the kernel applies a
   single, uniformly-shared scaling factor [inv] to every element of
   the input.  This factor is computed at runtime as
   [inv = rsqrt sumsq], where [sumsq] approximates the real-valued
   sum-of-squares of the input.  Because the device-side reduction is
   a tree-reduce in floating-point arithmetic, [sumsq] (and therefore
   [inv]) is *not* bit-exactly determined by the input; the kernel's
   postcondition therefore existentially binds [inv] (and [sumsq])
   while pinning the result to be exactly elementwise multiplication
   by that single scalar.

   The constraint
       [sumsq %~ frobenius_sumsq_r (to_real_seq input)]
   is what makes the spec genuinely about Frobenius normalisation
   rather than "any uniform scaling": it forces the existentially
   bound [sumsq] to approximate the mathematical
   [sum_j (real_of_input_j)^2].

   Note on the all-zero edge case: if the input is identically zero,
   the real sum-of-squares is [0], [rsqrt 0] in IEEE-754 yields
   [+inf], and the result is filled with [0 * +inf == NaN].  The spec
   above is satisfied: the existential [inv] takes the value
   [F32.rsqrt 0] (whatever it is), and the result is the
   corresponding NaN-filled sequence, which equals
   [lseq_map (smul_step inv) input] up to that NaN.  Callers wanting
   to forbid this case should add a precondition relating the input
   to a non-zero norm; we do not require it here, mirroring the host
   PyTorch reference (which also produces NaNs in that case). *)

open Kuiper.Scalars
open Kuiper.Real
open Kuiper.Approximates
module Seq = FStar.Seq
module KS = Kuiper.Seq.Common

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
