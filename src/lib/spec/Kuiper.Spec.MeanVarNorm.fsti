module Kuiper.Spec.MeanVarNorm

(* Functional specification for row-wise mean/variance normalisation
   (KernelBench L1 #34, #35):

       mean_r = (1/D) * Σ_j x[r,j]
       m2_r   = (1/D) * Σ_j x[r,j]^2
       var_r  = m2_r - mean_r^2
       inv_r  = 1 / sqrt(var_r + eps)
       y[r,j] = (x[r,j] - mean_r) * inv_r
              = inv_r * x[r,j] + (-mean_r * inv_r)

   This is the InstanceNorm/GroupNorm contract used by KernelBench
   #34/#35 (the bridge reshapes for #35).  As with the L1/L2 specs,
   each row's statistics are bound existentially because the
   device-side reductions are non-deterministic up to floating-point
   rounding and [rsqrt] / [div] are opaque to the spec.  We *can* pin:

     * shape: each row of the output equals an affine transform
       [a*x + b] applied to the corresponding input row;
     * value: [a == rsqrt(var_eps_r)], [b == -mean_r * a], with
       [mean_r], [m2_r] each approximating the real-valued
       (1/D)*Σ x and (1/D)*Σ x² respectively.

   We re-use [sq_step_r], [frobenius_sumsq_r] and [affine_result]
   from Kuiper.Spec.Frobenius.

   Edge cases: a constant row gives var = 0; the [eps] term keeps
   var_eps strictly positive for any reasonable [eps > 0], yielding
   a well-defined normalisation.  If [eps == 0] and the row is
   constant, [rsqrt 0 == +inf] and the row becomes NaN, matching the
   PyTorch reference. *)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
module Seq = FStar.Seq

(* Per-row predicate: row [r] of [sx'] is the mean/variance
   normalisation of the corresponding input row. *)
let row_mean_var_normalized
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#bd:nat)
  (sx sx' : Seq.lseq t bd)
  (off d : nat{off + d <= bd})
  (eps inv_d : t)
  : prop =
  exists (sum sum2 mean m2 var var_eps inv neg_mean_inv : t).
    (let row : Seq.lseq t d = Seq.slice sx off (off + d) in
     sum  %~ rsum (to_real_seq row) /\
     sum2 %~ frobenius_sumsq_r (to_real_seq row) /\
     mean         == mul sum  inv_d /\
     m2           == mul sum2 inv_d /\
     var          == sub m2 (mul mean mean) /\
     var_eps      == add var eps /\
     inv          == rsqrt var_eps /\
     neg_mean_inv == sub zero (mul mean inv) /\
     Seq.slice sx' off (off + d) ==
       affine_result #t inv neg_mean_inv #d row)

(* Whole-tensor spec: every row is mean/var-normalised. *)
let mean_var_post
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (b d : nat)
  (eps inv_d : t)
  (sx sx' : Seq.lseq t (b * d))
  : prop =
  forall (r : nat). r < b ==>
    (let lo : nat = r * d in
     let hi : nat = lo + d in
     hi <= b * d /\
     row_mean_var_normalized sx sx' lo d eps inv_d)
