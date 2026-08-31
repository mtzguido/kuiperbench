module Kuiper.Spec.KLDivLoss

(* Functional spec for KernelBench L1 #98:
   KL divergence with reduction='batchmean'.

   PyTorch reference:
     F.kl_div(torch.log(predictions), targets, reduction='batchmean')
     = sum_i targets[i] * (log(targets[i]) - log(predictions[i])) / batch_size

   The public kernel takes the positive batch size and returns the
   batch mean.  Its inputs are related to positive real sequences, so
   [log_approx] connects both floating logs to [Kuiper.Real.log]. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.KB.Compat.Map { lseq_map2 }
module Seq = FStar.Seq

inline_for_extraction
let kl_step (#t:Type0) {| scalar t, floating t |} (p tt : t) : t =
  mul tt (sub (flog tt) (flog p))

let real_kl_step (p tt : real) : real =
  if p >. 0.0R && tt >. 0.0R
  then tt *. (log tt -. log p)
  else 0.0R

let positive_seq
  (n : nat)
  (s : Seq.lseq real n)
  : prop =
  forall (i:nat). i < n ==> Seq.index s i >. 0.0R

let real_kl_sum
  (n : nat)
  (rp rt : Seq.lseq real n)
  : real =
  rsum (lseq_map2 real_kl_step rp rt)

let real_kl
  (n : nat)
  (batches : pos)
  (rp rt : Seq.lseq real n)
  : real =
  real_kl_sum n rp rt /. FStar.Real.of_int batches
