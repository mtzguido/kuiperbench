module Kuiper.KB.ReduceMean

(* KernelBench L1 #48: mean-reduction along dim=1 of a 3-D
   (B, D, M) row-major tensor; output shape (B, M).

   Pipeline (2 GPU launches):
     1. Reduce sum: y[b*M+j] %~ Σ_d x[b, d, j]
        via Kuiper.KB.ReduceSum.reduce_sum_fw_f32
     2. Scalar multiply: y[i] := y[i] * inv_d
        via Kuiper.KB.ScalarMul.smul_fw_f32

   The PyTorch reference is [torch.mean(x, dim=1)] which equals
   [torch.sum(x, dim=1) / D]; the bridge passes [inv_d = 1.0 / D]
   so the post is [sout' @! r == mul inv_d sumr] for some
   [sumr %~ rsum (real row)].

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.SumReduceDim
open Kuiper.Spec.MeanReduceDim
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT

(* Verified, extractable reciprocal 1/d as f32 (see .fst); the mean divisor is
   computed inside the verification boundary. *)
val reducemean_recip_f32 (d : szp) : f32

inline_for_extraction noextract
type reduce_mean_fw_ty (t:Type0) {| scalar t, real_like t |} =
  fn (b : szp)
     (m : szp { SZ.fits (SZ.v b * SZ.v m) /\ SZ.v b * SZ.v m <= max_blocks })
     (d : szp { SZ.v d <= max_threads /\
                SZ.fits (SZ.v d + max_threads) /\
                SZ.fits (SZ.v m * SZ.v d) /\
                SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) })
     (inv_d : t)
     (x : array2 t (l2_bcm_pages (SZ.v b) (SZ.v m) (SZ.v d)) { is_global x })
     (y : array1 t (l1_forward (SZ.v b * SZ.v m)) { is_global y })
     (#sx : erased (EM.chest2 t (SZ.v b * SZ.v m) (SZ.v d)))
     (#sy : erased (chest1 t (SZ.v b * SZ.v m)))
     requires
       cpu **
       on gpu_loc (x |-> sx) **
       on gpu_loc (y |-> sy)
     ensures
       cpu **
       on gpu_loc (x |-> sx) **
       (exists* (sy' : chest1 t (SZ.v b * SZ.v m)).
          on gpu_loc (y |-> sy') **
          pure (meanreduce_post (SZ.v b * SZ.v m) (SZ.v d) inv_d sx (chest1_to_seq sy')))

val reduce_mean_fw_f32 : reduce_mean_fw_ty f32
