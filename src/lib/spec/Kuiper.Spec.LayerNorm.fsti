module Kuiper.Spec.LayerNorm

(* Functional specification for row-wise LayerNorm
   (KernelBench L1 #40, with the (B, C, H, W) tensor viewed as (B, n)
   with n = C*H*W).  PyTorch's nn.LayerNorm(normalized_shape) reduces
   over the last len(normalized_shape) dims.  For each batch row r:

       sum_r   = Σ_j x[r,j]
       sumsq_r = Σ_j x[r,j]^2
       mean_r  = sum_r / n
       m2_r    = sumsq_r / n
       var_r   = m2_r - mean_r^2
       inv_r   = 1 / sqrt(var_r + eps)
       y[r,j]  = (x[r,j] - mean_r) * inv_r * γ[j] + β[j]

   The public contract directly approximates this real-valued operation;
   floating intermediates used by the implementation are not exposed as
   existential witnesses.  Its explicit domain requires each real
   [variance + eps] to be positive, exactly the premise needed by the
   temporary [Kuiper.KB.Compat.SqrtApprox.rsqrt_approx] law proposed
   for upstream Kuiper.
*)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
module Seq = FStar.Seq
module RealSqrt = FStar.Math.Sqrt

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

let ln_mean_r (#n:pos) (row:Seq.lseq real n) : real =
  rsum row *. (1.0R /. FStar.Real.of_int n)

let ln_m2_r (#n:pos) (row:Seq.lseq real n) : real =
  frobenius_sumsq_r row *. (1.0R /. FStar.Real.of_int n)

let ln_arg_r (#n:pos) (eps:real) (row:Seq.lseq real n) : real =
  ln_m2_r #n row -. ln_mean_r #n row *. ln_mean_r #n row +. eps

let ln_inv_r (#n:pos) (eps:real) (row:Seq.lseq real n) : real =
  let a = ln_arg_r #n eps row in
  if a >. 0.0R then RealSqrt.rsqrt a else 0.0R

let ln_row_result_r
  (#n:pos) (eps:real)
  (gamma beta row:Seq.lseq real n) : Seq.lseq real n =
  let mean = ln_mean_r #n row in
  let inv = ln_inv_r #n eps row in
  Seq.init n (fun j ->
    let normalized = Seq.index row j *. inv +. (0.0R -. mean *. inv) in
    normalized *. Seq.index gamma j +. Seq.index beta j)

let row_layernorm_domain
  (#t:Type0) {| scalar t, real_like t |}
  (#bn:nat) (sx:Seq.lseq t bn)
  (off:nat) (n:pos{off+n <= bn}) (eps:t) : prop =
  ln_arg_r #n (to_real eps)
    (to_real_seq (Seq.slice sx off (off+n))) >. 0.0R

(* Direct real-valued per-row LayerNorm contract. *)
let row_layer_normalized
  (#t:Type0) {| scalar t, real_like t |}
  (#bn:nat) (#n:pos)
  (sx sx' : Seq.lseq t bn)
  (gamma beta : Seq.lseq t n)
  (off : nat{off + n <= bn})
  (eps : t) : prop =
  Seq.slice sx' off (off+n) %~
    ln_row_result_r #n (to_real eps)
      (to_real_seq gamma) (to_real_seq beta)
      (to_real_seq (Seq.slice sx off (off+n)))

let layernorm_domain
  (#t:Type0) {| scalar t, real_like t |}
  (b:nat) (n:pos) (eps:t) (sx:Seq.lseq t (b*n)) : prop =
  forall (r:nat). r < b ==>
    row_layernorm_domain sx (r*n) n eps

(* Whole-tensor spec: every row is LayerNorm-normalised. *)
let layernorm_post
  (#t:Type0) {| scalar t, real_like t |}
  (b : nat) (n:pos)
  (eps : t)
  (gamma beta : Seq.lseq t n)
  (sx sx' : Seq.lseq t (b * n))
  : prop =
  forall (r : nat). r < b ==>
    (let lo : nat = r * n in
     let hi : nat = lo + n in
     hi <= b * n /\
     row_layer_normalized sx sx' gamma beta lo eps)
