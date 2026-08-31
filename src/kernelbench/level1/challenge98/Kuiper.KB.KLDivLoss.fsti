module Kuiper.KB.KLDivLoss

(* KernelBench helper for L1 #98 (KL divergence, batchmean).

   Pipeline (2 GPU launches):
     1. [map_gpu2] elementwise: predictions[i] := t[i] * (log t[i] - log p[i])
     2. [HRed.reduce] (identity pre-map) → scalar sum
     3. verified division by [batch_size] → batch mean

   The verified entry point also divides by [batch_size], and its result
   directly approximates [Kuiper.Spec.KLDivLoss.real_kl]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.KLDivLoss

inline_for_extraction noextract
type kl_fw_ty (t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |} =
  fn
    (n : szp {n <= max_blocks * max_threads})
    (batches : szp)
    (predictions : array1 t (l1_forward n) { is_global predictions })
    (targets     : array1 t (l1_forward n) { is_global targets })
    (#sp #st : chest1 t n)
    (rp rt : erased (lseq real n))
    (#fb : perm)
    preserves
      cpu ** on gpu_loc (targets |-> Frac fb st) **
      pure (sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt)
    requires
      on gpu_loc (predictions |-> sp) **
      pure (positive_seq n rp /\ positive_seq n rt)
    returns
      res : t
    ensures
      (exists* (sp' : chest1 t n).
         on gpu_loc (predictions |-> sp') **
         pure (res %~ real_kl n batches rp rt))

val kl_div_fw_f32 : kl_fw_ty f32
