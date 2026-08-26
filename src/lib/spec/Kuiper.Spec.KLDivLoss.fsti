module Kuiper.Spec.KLDivLoss

(* Functional spec for KernelBench L1 #98:
   KL divergence with reduction='batchmean'.

   PyTorch reference:
     F.kl_div(torch.log(predictions), targets, reduction='batchmean')
     = sum_i targets[i] * (log(targets[i]) - log(predictions[i])) / batch_size

   The kernel returns the *unscaled* sum (the kernel cannot in general
   know the batch size from the flat inputs).  The host divides by
   [batch_size] after the kernel returns.

   Because [rlog] is partial in [Kuiper.Real], we keep the elementwise
   step at the floating-point level here and pin the kernel's result
   to the [%~]-approximation of the f32 step's real-image sum. *)

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

let real_kl_sum
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (n : nat)
  (sp st : Seq.lseq t n)
  : real
  = rsum (to_real_seq (lseq_map2 (kl_step #t) sp st))
