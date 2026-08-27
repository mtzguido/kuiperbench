module Kuiper.KB.KLDivLoss

(* KernelBench helper for L1 #98 (KL divergence, batchmean).

   Pipeline (2 GPU launches):
     1. [map_gpu2] elementwise: predictions[i] := t[i] * (log t[i] - log p[i])
     2. [HRed.reduce] (identity pre-map) → host scalar [s] (the *unscaled* sum)

   The host divides the returned [s] by [batch_size] to obtain the final
   batchmean.

   Spec: [Kuiper.Spec.KLDivLoss.real_kl_sum]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.KLDivLoss
(* Verified, extractable batchmean reduction (see .fst). *)
val kl_div_mean_f32 (s : f32) (b : szp) : f32

inline_for_extraction noextract
type kl_fw_ty (t:Type0) {| scalar t, real_like t, floating t |} =
  fn
    (n : szp {n <= max_blocks * max_threads})
    (predictions : array1 t (l1_forward n) { is_global predictions })
    (targets     : array1 t (l1_forward n) { is_global targets })
    (#sp #st : chest1 t n)
    (#fb : perm)
    preserves
      cpu ** on gpu_loc (targets |-> Frac fb st)
    requires
      on gpu_loc (predictions |-> sp)
    returns
      res : t
    ensures
      (exists* (sp' : chest1 t n).
         on gpu_loc (predictions |-> sp') **
         pure (res %~ real_kl_sum n (chest1_to_seq sp) (chest1_to_seq st)))

val kl_div_fw_f32 : kl_fw_ty f32
