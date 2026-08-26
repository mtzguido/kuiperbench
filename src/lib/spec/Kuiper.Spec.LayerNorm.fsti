module Kuiper.Spec.LayerNorm

(* Functional specification for row-wise LayerNorm
   (KernelBench L1 #40, with the (B, C, H, W) tensor viewed as (B, n)
   with n = C*H*W).  PyTorch's nn.LayerNorm(normalized_shape) reduces
   over the last len(normalized_shape) dims.  For each batch row r:

       sum_r   = Σ_j x[r,j]
       sumsq_r = Σ_j x[r,j]^2
       mean_r  = sum_r * inv_n              -- inv_n ≈ 1/n
       m2_r    = sumsq_r * inv_n
       var_r   = m2_r - mean_r^2
       inv_r   = 1 / sqrt(var_r + eps)
       y[r,j]  = (x[r,j] - mean_r) * inv_r * γ[j] + β[j]

   The spec mirrors MeanVarNorm but adds the per-element column
   broadcast of γ (scale) and β (shift), of length n.  The per-row
   statistics (sum, sumsq, mean, m2, var, var_eps, inv) are
   existentially bound exactly as in MeanVarNorm.

   Edge cases match PyTorch:
     - Constant row: var = 0, inv = rsqrt(eps); finite for eps > 0.
     - eps = 0: rsqrt 0 = +inf, output = NaN.  Spec is satisfied:
       the existentially-bound [inv] is whatever rsqrt produces.
*)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
module Seq = FStar.Seq

(* Per-element body of LayerNorm given the per-row scalars
   (inv, neg_mean_inv) and per-column γ[j], β[j].
       y = (inv * x + neg_mean_inv) * γ + β
   i.e. ((x - mean)/std) * γ + β. *)
inline_for_extraction
let ln_step
  (#t:Type0) {| scalar t |}
  (inv neg_mean_inv : t) (g b : t) (x : t) : t =
  add (mul (add (mul x inv) neg_mean_inv) g) b

(* Build the resulting row from γ, β, the per-row affine scalars and
   the input row. *)
let ln_row_result
  (#t:Type0) {| scalar t |}
  (#n:nat)
  (inv neg_mean_inv : t)
  (gamma beta : Seq.lseq t n)
  (row : Seq.lseq t n)
  : GTot (Seq.lseq t n) =
  Seq.init_ghost n (fun j ->
    ln_step inv neg_mean_inv (Seq.index gamma j) (Seq.index beta j) (Seq.index row j))

(* Per-row predicate: row [r] of [sx'] is the LayerNorm-normalised
   version of row [r] of [sx], existentially binding all the
   floating-point per-row statistics. *)
let row_layer_normalized
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#bn:nat) (#n:nat)
  (sx sx' : Seq.lseq t bn)
  (gamma beta : Seq.lseq t n)
  (off : nat{off + n <= bn})
  (eps inv_n : t)
  : prop =
  exists (sum sumsq mean m2 var var_eps inv neg_mean_inv : t).
    (let row : Seq.lseq t n = Seq.slice sx off (off + n) in
     sum   %~ rsum (to_real_seq row) /\
     sumsq %~ frobenius_sumsq_r (to_real_seq row) /\
     mean         == mul sum   inv_n /\
     m2           == mul sumsq inv_n /\
     var          == sub m2 (mul mean mean) /\
     var_eps      == add var eps /\
     inv          == rsqrt var_eps /\
     neg_mean_inv == sub Kuiper.Scalars.zero (mul mean inv) /\
     Seq.slice sx' off (off + n) ==
       ln_row_result inv neg_mean_inv gamma beta row)

(* Whole-tensor spec: every row is LayerNorm-normalised. *)
let layernorm_post
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (b n : nat)
  (eps inv_n : t)
  (gamma beta : Seq.lseq t n)
  (sx sx' : Seq.lseq t (b * n))
  : prop =
  forall (r : nat). r < b ==>
    (let lo : nat = r * n in
     let hi : nat = lo + n in
     hi <= b * n /\
     row_layer_normalized sx sx' gamma beta lo eps inv_n)
