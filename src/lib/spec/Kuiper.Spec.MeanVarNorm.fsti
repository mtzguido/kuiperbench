module Kuiper.Spec.MeanVarNorm

(* Functional specification for row-wise mean/variance normalisation
   (KernelBench L1 #34, #35):

       mean_r = (1/D) * Σ_j x[r,j]
       m2_r   = (1/D) * Σ_j x[r,j]^2
       var_r  = m2_r - mean_r^2
       inv_r  = 1 / sqrt(var_r + eps)
       y[r,j] = (x[r,j] - mean_r) * inv_r
              = inv_r * x[r,j] + (-mean_r * inv_r)

   The public postcondition directly relates every output element to
   this real normalization.  Floating reductions and affine scalars are
   proof-local and are not existentially exposed.

   We re-use [sq_step_r], [frobenius_sumsq_r] and [affine_result]
   from Kuiper.Spec.Frobenius.

   The explicit domain requires each real [variance + eps] to be
   positive, which is the premise of the temporary reciprocal-square-root
   approximation law proposed for upstream Kuiper. *)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
module Seq = FStar.Seq
module RealSqrt = FStar.Math.Sqrt

let mvn_mean_r (#d:pos) (row:Seq.lseq real d) : real =
  rsum row *. (1.0R /. FStar.Real.of_int d)

let mvn_m2_r (#d:pos) (row:Seq.lseq real d) : real =
  frobenius_sumsq_r row *. (1.0R /. FStar.Real.of_int d)

let mvn_arg_r (#d:pos) (eps:real) (row:Seq.lseq real d) : real =
  mvn_m2_r #d row -. mvn_mean_r #d row *. mvn_mean_r #d row +. eps

let mvn_inv_r (#d:pos) (eps:real) (row:Seq.lseq real d) : real =
  let a = mvn_arg_r #d eps row in
  if a >. 0.0R then RealSqrt.rsqrt a else 0.0R

let mvn_row_result_r (#d:pos) (eps:real) (row:Seq.lseq real d)
  : Seq.lseq real d =
  let mean = mvn_mean_r #d row in
  let inv = mvn_inv_r #d eps row in
  Seq.init d (fun j ->
    Seq.index row j *. inv +. (0.0R -. mean *. inv))

let row_mean_var_domain
  (#t:Type0) {| scalar t, real_like t |}
  (#bd:nat) (sx:Seq.lseq t bd)
  (off:nat) (d:pos{off+d <= bd}) (eps:t) : prop =
  mvn_arg_r #d (to_real eps)
    (to_real_seq (Seq.slice sx off (off+d))) >. 0.0R

let row_mean_var_normalized
  (#t:Type0) {| scalar t, real_like t |}
  (#bd:nat) (sx sx':Seq.lseq t bd)
  (off:nat) (d:pos{off+d <= bd}) (eps:t) : prop =
  Seq.slice sx' off (off+d) %~
    mvn_row_result_r #d (to_real eps)
      (to_real_seq (Seq.slice sx off (off+d)))

(* Whole-tensor spec: every row is mean/var-normalised. *)
let mean_var_domain
  (#t:Type0) {| scalar t, real_like t |}
  (b:nat) (d:pos) (eps:t)
  (sx:Seq.lseq t (b*d)) : prop =
  forall (r:nat). r < b ==>
    row_mean_var_domain sx (r*d) d eps

let mean_var_post
  (#t:Type0) {| scalar t, real_like t |}
  (b:nat) (d:pos)
  (eps : t)
  (sx sx' : Seq.lseq t (b * d))
  : prop =
  forall (r : nat). r < b ==>
    (let lo : nat = r * d in
     let hi : nat = lo + d in
     hi <= b * d /\
     row_mean_var_normalized sx sx' lo d eps)
