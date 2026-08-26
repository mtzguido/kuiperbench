module Kuiper.Spec.BatchNorm

(* Functional specification for per-channel BatchNorm
   (KernelBench L1 #33).  PyTorch's nn.BatchNorm2d in default training
   mode uses *batch* statistics for the forward normalization, with
   biased variance (denom = N*H*W, not N*H*W - 1).  Viewing the
   physical (N, C, H, W) tensor as a logical (C, N*HW) matrix, for
   every channel ci:

       sum_ci   = Σ_k x[ci,k]
       sumsq_ci = Σ_k x[ci,k]^2
       mean_ci  = sum_ci   * inv_n      -- inv_n ≈ 1/(N*H*W)
       m2_ci    = sumsq_ci * inv_n
       var_ci   = m2_ci - mean_ci^2     -- biased variance
       inv_ci   = 1 / sqrt(var_ci + eps)
       y[ci,k]  = (x[ci,k] - mean_ci) * inv_ci * γ[ci] + β[ci]

   Mirror of Kuiper.Spec.LayerNorm but with per-row (rather than
   per-column) γ, β scalars: γ : lseq t c, β : lseq t c.

   The per-channel statistics (sum, sumsq, mean, m2, var, var_eps,
   inv, neg_mean_inv) are existentially bound exactly as in LayerNorm.

   Edge cases match PyTorch:
     - Constant channel: var = 0, inv = rsqrt(eps); finite for eps > 0.
     - eps = 0: rsqrt 0 = +inf, output = NaN.  Spec is satisfied:
       the existentially-bound [inv] is whatever rsqrt produces.

   Note: nn.BatchNorm2d also mutates running_mean/running_var as a
   side effect in training mode.  KernelBench's eval_kernel_against_ref
   only compares output tensors, so we do *not* model the running
   statistics here.  *)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
open Kuiper.EMatrix
module Seq = FStar.Seq

(* Per-element body of BatchNorm given the per-channel scalars
   (inv, neg_mean_inv, γ, β):
       y = (inv * x + neg_mean_inv) * γ + β
   i.e. ((x - mean)/std) * γ + β.  Note this is identical to
   Kuiper.Spec.LayerNorm.ln_step; we re-define here for module
   independence. *)
inline_for_extraction
let bn_step
  (#t:Type0) {| scalar t |}
  (inv neg_mean_inv g b : t) (x : t) : t =
  add (mul (add (mul x inv) neg_mean_inv) g) b

(* Build the resulting channel from γ[ci], β[ci], the per-channel
   affine scalars and the input channel data. *)
let bn_row_result
  (#t:Type0) {| scalar t |}
  (#nhw:nat)
  (inv neg_mean_inv g b : t)
  (row : Seq.lseq t nhw)
  : GTot (Seq.lseq t nhw) =
  Seq.init_ghost nhw (fun k ->
    bn_step inv neg_mean_inv g b (Seq.index row k))

(* Per-channel predicate: channel [ci] of [sx'] is the BatchNorm-
   normalised version of channel [ci] of [sx], existentially binding
   all the floating-point per-channel statistics.  [sx], [sx'] are
   ematrices over (c, n*hw); γ, β are per-channel scalars. *)
let row_batch_normalized
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#c #nhw : nat)
  (sx sx' : chest2 t c nhw)
  (gamma beta : Seq.lseq t c)
  (ci : nat{ci < c})
  (eps inv_n : t)
  : prop =
  exists (sum sumsq mean m2 var var_eps inv neg_mean_inv : t).
    (let row : Seq.lseq t nhw = ematrix_row sx ci in
     sum   %~ rsum (to_real_seq row) /\
     sumsq %~ frobenius_sumsq_r (to_real_seq row) /\
     mean         == mul sum   inv_n /\
     m2           == mul sumsq inv_n /\
     var          == sub m2 (mul mean mean) /\
     var_eps      == add var eps /\
     inv          == rsqrt var_eps /\
     neg_mean_inv == sub Kuiper.Scalars.zero (mul mean inv) /\
     ematrix_row sx' ci ==
       bn_row_result inv neg_mean_inv
         (Seq.index gamma ci) (Seq.index beta ci) row)

(* Whole-tensor spec: every channel is BatchNorm-normalised. *)
let batchnorm_post
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (c nhw : nat)
  (eps inv_n : t)
  (gamma beta : Seq.lseq t c)
  (sx sx' : chest2 t c nhw)
  : prop =
  forall (ci : nat). ci < c ==>
    row_batch_normalized sx sx' gamma beta ci eps inv_n
