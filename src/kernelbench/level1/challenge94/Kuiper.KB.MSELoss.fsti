module Kuiper.KB.MSELoss

(* KernelBench L1 #94 (Mean Squared Error).

   The public entry preserves both original inputs, performs the complete
   pipeline in private Kuiper-owned scratch, and returns an owned one-element
   GPU buffer:

   Pipeline:
     1. [map_gpu2] elementwise: predictions[i] := (p[i] - t[i])^2
     2. [HRed.reduce] (identity pre-map) → Kuiper-local scalar [s]
     3. verified division by [n]
     4. verified write to the returned GPU scalar

   Spec: direct approximation of [real_mse]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.MSELoss

fn mse_loss_fw_f32
  (n : szp {n <= max_blocks * max_threads})
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
  returns out : array1 f32 (l1_forward 1)
  ensures
    exists* (sout : chest1 f32 1).
      on gpu_loc (out |-> sout) **
      pure (acc1 sout 0 %~ real_mse n rp rt)
