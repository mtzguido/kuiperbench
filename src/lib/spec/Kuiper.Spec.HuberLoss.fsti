module Kuiper.Spec.HuberLoss

(* Functional specification for KernelBench L1 #96: Smooth L1 (Huber)
   Loss with the PyTorch defaults (beta = 1, reduction = 'mean').

   PyTorch reference:
       torch.nn.functional.smooth_l1_loss(predictions, targets)
     = mean_i  smooth_l1_step (predictions_i - targets_i)
   where
       smooth_l1_step x = 0.5 * x^2     if |x| < 1
                       = |x| - 0.5      otherwise.

   We elect to make this spec f32-specific (rather than polymorphic
   over [floating t]) because the constants 0.5 and 1.0 are most
   cleanly written as f32 literals.

   Same [%~] approximation pattern as the other losses: the
   elementwise step is bit-exact f32; only the tree-reduction is
   approximated against the real sum. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.KB.Compat.Map { lseq_map2 }

let rabs (x : real) : real =
  if x <. 0.0R then 0.0R -. x else x

let real_huber_step (a b : real) : real =
  let d = a -. b in
  let ad = rabs d in
  if ad <. 1.0R
  then (d *. d) /. 2.0R
  else ad -. 0.5R

let real_huber
  (n : pos)
  (ra rb : Seq.lseq real n)
  : real
  = rsum (lseq_map2 real_huber_step ra rb) /. FStar.Real.of_int n
