module Kuiper.Spec.BatchNorm

open Kuiper.Real
open Kuiper.EMatrix
open Kuiper.Spec.Frobenius
module Seq = FStar.Seq
module RealSqrt = FStar.Math.Sqrt

let bn_row_var_eps_positive
  (#c #nhw : nat)
  (eps inv_n : real)
  (sx : chest2 real c nhw)
  (ci : nat{ci < c})
  = ()

let real_bn_row_result
  (#c #nhw : nat)
  (eps inv_n : real)
  (sx : chest2 real c nhw { batchnorm_domain c nhw eps inv_n sx })
  (gamma beta : Seq.lseq real c)
  (ci : nat{ci < c})
  = let row = ematrix_row sx ci in
    let mean = bn_row_mean inv_n row in
    let var_eps = bn_row_var_eps eps inv_n row in
    bn_row_var_eps_positive eps inv_n sx ci;
    let positive_var_eps : RealSqrt.rpos = var_eps in
    let inv = RealSqrt.rsqrt positive_var_eps in
    Seq.init_ghost nhw (fun k ->
      (((Seq.index row k *. inv) +. (0.0R -. (mean *. inv))) *.
        Seq.index gamma ci) +. Seq.index beta ci)

let real_bn_row_result_unfold
  (#c #nhw : nat)
  (eps inv_n : real)
  (sx : chest2 real c nhw { batchnorm_domain c nhw eps inv_n sx })
  (gamma beta : Seq.lseq real c)
  (ci : nat{ci < c})
  = ()
