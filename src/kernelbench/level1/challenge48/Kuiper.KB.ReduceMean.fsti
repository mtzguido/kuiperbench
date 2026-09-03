module Kuiper.KB.ReduceMean

(* KernelBench L1 #48: mean-reduction along dim=1 of a 3-D
   (B, D, M) row-major tensor; output shape (B, M).

   Pipeline (2 GPU launches):
     1. Reduce sum: y[b*M+j] %~ Σ_d x[b, d, j]
        via Kuiper.KB.ReduceSum.reduce_sum_fw_f32
     2. Scalar multiply: y[i] := y[i] * inv_d
        via Kuiper.KB.ScalarMul.smul_fw_f32

   The PyTorch reference is [torch.mean(x, dim=1)].  The verified entry
   computes [1/D] from [d] itself, and its post directly approximates
   the real row mean; no floating-point reduction witness escapes.

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

fn reduce_mean_fw_f32
  (b : szp)
  (m : szp { SZ.fits (SZ.v b * SZ.v m) /\ SZ.v b * SZ.v m <= max_blocks })
  (d : szp { SZ.fits (SZ.v d + max_threads) /\
             SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (y : array1 f32 (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  (#sy : chest1 f32 (SZ.v b * SZ.v m))
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (y |-> sy)
  ensures
    (exists* (sy' : chest1 f32 (SZ.v b * SZ.v m)).
       on gpu_loc (y |-> sy') **
       pure (meanreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy')))

fn reduce_mean_alloc_f32
  (b : szp)
  (m : szp { SZ.fits (SZ.v b * SZ.v m) /\ SZ.v b * SZ.v m <= max_blocks })
  (d : szp { SZ.fits (SZ.v d + max_threads) /\
             SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  norewrite
  preserves cpu ** on gpu_loc (x |-> sx)
  returns y : array1 f32 (l1_forward (SZ.v b * SZ.v m))
  ensures
    exists* (sy : chest1 f32 (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy) **
      pure (meanreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy))
