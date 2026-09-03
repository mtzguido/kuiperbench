module Kuiper.KB.HuberLoss

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.HuberLoss

(* Complete self-allocating Smooth-L1 entry.  Both public inputs are
   preserved; pointwise work happens in verified private scratch, and the
   returned one-element GPU buffer approximates the real mean loss. *)
fn huber_loss_fw_f32
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
      pure (acc1 sout 0 %~ real_huber n rp rt)
