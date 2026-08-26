module Kuiper.Spec.MSELoss

open Kuiper.Common
open Kuiper.Real
open Kuiper.Approximates
open Kuiper.KB.Compat.Map { lseq_map2 }

let real_mse_step (a b : real) : real =
  let d = a -. b in
  d *. d

let real_mse
  (n : pos)
  (ra rb : Seq.lseq real n)
  : real
  = rsum (lseq_map2 real_mse_step ra rb) /. FStar.Real.of_int n
