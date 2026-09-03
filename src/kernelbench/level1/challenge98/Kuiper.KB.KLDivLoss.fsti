module Kuiper.KB.KLDivLoss

(* KernelBench L1 #98 (KL divergence, batchmean).

   Pipeline (2 GPU launches):
     1. [map_gpu2] elementwise: predictions[i] := t[i] * (log t[i] - log p[i])
     2. [HRed.reduce] (identity pre-map) → scalar sum
     3. verified division by [batch_size] → batch mean

   The public entry preserves both inputs by using private scratch, divides by
   [batch_size], and returns an owned one-element GPU buffer that directly
   approximates [Kuiper.Spec.KLDivLoss.real_kl]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.KLDivLoss

fn kl_div_fw_f32
  (n : szp {n <= max_blocks * max_threads})
  (batches : szp)
  (predictions : array1 f32 (l1_forward n) { is_global predictions })
  (targets     : array1 f32 (l1_forward n) { is_global targets })
  (#sp #st : chest1 f32 n)
  (rp rt : erased (lseq real n))
  (#fp #ft : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (predictions |-> Frac fp sp) **
    on gpu_loc (targets |-> Frac ft st) **
    pure (sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt)
  requires
    pure (positive_seq n rp /\ positive_seq n rt)
  returns out : array1 f32 (l1_forward 1)
  ensures
    exists* (sout : chest1 f32 1).
      on gpu_loc (out |-> sout) **
      pure (acc1 sout 0 %~ real_kl n batches rp rt)
