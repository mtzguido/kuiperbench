module Kuiper.Spec.BatchNorm

(* Direct real-valued functional specification for KernelBench L1 #33.

   The physical NCHW input is exposed to the kernel as a logical C x (N*HW)
   matrix.  For every channel this specification computes the real sum,
   sum-of-squares, biased variance, inverse standard deviation, and affine
   output.  Unlike the former contract, none of those deterministic
   intermediates is existentially quantified. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Approximates
open Kuiper.Spec.Frobenius
open Kuiper.EMatrix
module Seq = FStar.Seq
module RealSqrt = FStar.Math.Sqrt
unfold let f32 = Kuiper.Float32.t

inline_for_extraction
let bn_step
  (#t:Type0) {| scalar t |}
  (inv neg_mean_inv g b : t) (x : t) : t =
  add (mul (add (mul x inv) neg_mean_inv) g) b

let bn_row_result
  (#t:Type0) {| scalar t |}
  (#nhw:nat)
  (inv neg_mean_inv g b : t)
  (row : Seq.lseq t nhw)
  : GTot (Seq.lseq t nhw) =
  Seq.init_ghost nhw (fun k ->
    bn_step inv neg_mean_inv g b (Seq.index row k))

let bn_row_sum (#nhw:nat) (row : Seq.lseq real nhw) : real =
  rsum row

let bn_row_sumsq (#nhw:nat) (row : Seq.lseq real nhw) : real =
  frobenius_sumsq_r row

let bn_row_mean (#nhw:nat) (inv_n : real) (row : Seq.lseq real nhw) : real =
  bn_row_sum row *. inv_n

let bn_row_var (#nhw:nat) (inv_n : real) (row : Seq.lseq real nhw) : real =
  bn_row_sumsq row *. inv_n -.
    bn_row_mean inv_n row *. bn_row_mean inv_n row

let bn_row_var_eps
  (#nhw:nat) (eps inv_n : real) (row : Seq.lseq real nhw) : real =
  bn_row_var inv_n row +. eps

let bn_inv_n_r (nhw : pos) : real =
  1.0R /. FStar.Real.of_int nhw

(* [FStar.Math.Sqrt.rsqrt] is intentionally partial at zero.  KernelBench's
   BatchNorm inputs use positive epsilon; stating the exact semantic domain
   also rules out pretending that an invalid inverse square root has a real
   result. *)
let batchnorm_domain
  (c nhw : nat)
  (eps inv_n : real)
  (sx : chest2 real c nhw)
  : prop =
  forall (ci : nat). ci < c ==>
    bn_row_var_eps eps inv_n (ematrix_row sx ci) >. 0.0R

val bn_row_var_eps_positive
  (#c #nhw : nat)
  (eps inv_n : real)
  (sx : chest2 real c nhw)
  (ci : nat{ci < c})
  : Lemma
      (requires batchnorm_domain c nhw eps inv_n sx)
      (ensures bn_row_var_eps eps inv_n (ematrix_row sx ci) >. 0.0R)

val real_bn_row_result
  (#c #nhw : nat)
  (eps inv_n : real)
  (sx : chest2 real c nhw { batchnorm_domain c nhw eps inv_n sx })
  (gamma beta : Seq.lseq real c)
  (ci : nat{ci < c})
  : GTot (Seq.lseq real nhw)

val real_bn_row_result_unfold
  (#c #nhw : nat)
  (eps inv_n : real)
  (sx : chest2 real c nhw { batchnorm_domain c nhw eps inv_n sx })
  (gamma beta : Seq.lseq real c)
  (ci : nat{ci < c})
  : Lemma
      (real_bn_row_result eps inv_n sx gamma beta ci ==
       (let row = ematrix_row sx ci in
       let mean = bn_row_mean inv_n row in
       let var_eps = bn_row_var_eps eps inv_n row in
       let positive_var_eps : RealSqrt.rpos = var_eps in
       let inv = RealSqrt.rsqrt positive_var_eps in
       Seq.init_ghost nhw (fun k ->
         (((Seq.index row k *. inv) +. (0.0R -. (mean *. inv))) *.
           Seq.index gamma ci) +. Seq.index beta ci)))

let row_batch_normalized
  (#c #nhw : nat)
  (eps inv_n : real)
  (sx : chest2 real c nhw { batchnorm_domain c nhw eps inv_n sx })
  (gamma beta : Seq.lseq real c)
  (sx' : chest2 f32 c nhw)
  (ci : nat{ci < c})
  : prop =
  ematrix_row sx' ci %~ real_bn_row_result eps inv_n sx gamma beta ci

let batchnorm_post
  (c nhw : nat)
  (eps inv_n : real)
  (gamma beta : Seq.lseq real c)
  (sx : chest2 real c nhw { batchnorm_domain c nhw eps inv_n sx })
  (sx' : chest2 f32 c nhw)
  : prop =
  forall (ci : nat). ci < c ==>
    row_batch_normalized eps inv_n sx gamma beta sx' ci
