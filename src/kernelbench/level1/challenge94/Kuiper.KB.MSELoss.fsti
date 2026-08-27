module Kuiper.KB.MSELoss

(* KernelBench helper for L1 #94 (Mean Squared Error).

   Pipeline (2 GPU launches):
     1. [map_gpu2] elementwise: predictions[i] := (p[i] - t[i])^2
     2. [HRed.reduce] (identity pre-map) → host scalar [s]

   Then just divide s by n on the host to get the final MSE.
   Spec: [Kuiper.Spec.MSELoss.mse_post]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.MSELoss
inline_for_extraction noextract
type mse_fw_ty (t:Type0) {| scalar t, real_like t |} =
  fn
    (n : szp {n <= max_blocks * max_threads})
    (predictions : array1 t (l1_forward n) { is_global predictions })
    (targets     : array1 t (l1_forward n) { is_global targets })
    (#sp #st : chest1 t n)
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
         pure (res %~ real_mse n rp rt))

val mse_loss_fw_f32 : mse_fw_ty f32
