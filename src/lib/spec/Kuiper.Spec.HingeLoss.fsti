module Kuiper.Spec.HingeLoss

(* Functional spec for KernelBench L1 #100: Hinge Loss.

       L = mean_i  max(0, 1 - p_i * t_i)

   Real-valued spec following the MSELoss template (real_mse_step /
   real_mse): a pure-[real] step and a pure-[real] mean.  The kernel
   then ensures [res %~ real_hinge n rp rt] where [rp]/[rt] are the
   real-valued ghost approximations of the f32 inputs. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.KB.Compat.Map { lseq_map2 }

let real_hinge_step (p t : real) : real =
  rmax 0.0R (1.0R -. p *. t)

let real_hinge
  (n : pos)
  (rp rt : Seq.lseq real n)
  : real
  = rsum (lseq_map2 real_hinge_step rp rt) /. FStar.Real.of_int n
