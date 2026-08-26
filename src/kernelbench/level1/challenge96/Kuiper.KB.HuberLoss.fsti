module Kuiper.KB.HuberLoss

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.HuberLoss
inline_for_extraction noextract
type huber_fw_ty (t:Type0) {| scalar t, real_like t |} =
  fn
    (n : szp {n <= max_blocks * max_threads})
    (predictions : array1 t (l1_forward n) { is_global predictions })
    (targets     : array1 t (l1_forward n) { is_global targets })
    (#sp #st : erased (chest1 t n))
    (rp  rt  : erased (lseq real n))
    (#fb : perm)
    preserves
      cpu ** on gpu_loc (targets |-> Frac fb st) **
      pure (sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt)
    requires
      on gpu_loc (predictions |-> sp)
    returns
      res : t
    ensures
      (exists* (sp' : chest1 t n).
         on gpu_loc (predictions |-> sp') **
         pure (res %~ real_huber n rp rt))

val huber_loss_fw_f32 : huber_fw_ty f32
