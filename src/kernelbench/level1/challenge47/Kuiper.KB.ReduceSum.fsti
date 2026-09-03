module Kuiper.KB.ReduceSum

(* KernelBench L1 #47: sum reduction over the middle dimension of a
   (B, D, M) row-major tensor.  PyTorch reference:
       y = torch.sum(x, dim=1, keepdim=True)    # shape (B, 1, M)

   Kuiper view: factor the (B, D, M) buffer as a 2-D matrix of shape
   (B*M, D) using [Kuiper.Tensor.Layout.BCMPages.l2_bcm_pages] with
   parameters (b=B, hw=M, c=D).  Row r = b*M + j carries the length-D
   slice x[b, :, j].  A single launch of the block-per-row tree
   reduction primitive [Kuiper.Kernel.HReduce.Block.reduce_batched_block]
   produces, per row r, an f32 that approximates the real-arithmetic
   sum of that row (i.e., y[b, 0, j]).

   The output is a flat length-(B*M) Array1; the bridge reshapes it
   to (B, 1, M) on the host for the harness comparison.

   Exactly 1 GPU kernel launch.  No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.SumReduceDim
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT

fn reduce_sum_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks /\
             SZ.fits (SZ.v d + max_threads) })
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
       pure (sumreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy')))

fn reduce_sum_alloc_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks /\
             SZ.fits (SZ.v d + max_threads) })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  norewrite
  preserves cpu ** on gpu_loc (x |-> sx)
  returns y : array1 f32 (l1_forward (SZ.v b * SZ.v m))
  ensures
    exists* (sy : chest1 f32 (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy) **
      pure (sumreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy))
