module Kuiper.KB.MaxReduceDim

(* KernelBench L1 #49: max reduction over the middle dimension of a
   (B, D, M) row-major tensor.  PyTorch reference:
       y = torch.max(x, dim=1).values      # shape (B, M)

   Kuiper view: factor the (B, D, M) buffer as a 2-D matrix of shape
   (B*M, D) using [Kuiper.Tensor.Layout.BCMPages.l2_bcm_pages] with
   parameters (b=B, hw=M, c=D).  Row r = b*M + j carries the length-D
   slice x[b, :, j].  A single launch of the one-block-per-row
   primitive [Kuiper.Kernel.HReduce.Block.Max.reduce_batched_block_max]
   produces, per row r, an f32 that approximates the real-arithmetic max
   of that row (i.e., y[b, j]).

   The block tree-reduction order in floating point is not bit-
   determined, so the post is stated up to the [%~] approximation
   relation (mirroring [Kuiper.Spec.SumReduceDim]).

   The output is a flat length-(B*M) Array1; the bridge reshapes it
   to (B, M) on the host for the harness comparison.

   Exactly 1 GPU kernel launch.  No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.MaxReduceDim
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq

fn maxreduce_dim_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks /\
             SZ.fits (SZ.v d + max_threads) })
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v m) (SZ.v d)) { is_global x })
  (y : array1 f32 (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : EM.chest2 f32 (SZ.v b * SZ.v m) (SZ.v d))
  (#sy : chest1 f32 (SZ.v b * SZ.v m))
  preserves cpu ** on gpu_loc (x |-> sx)
  requires
    on gpu_loc (y |-> sy)
  ensures
    exists* (sy' : chest1 f32 (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy') **
      pure (maxreduce_post (SZ.v b * SZ.v m) (SZ.v d) sx (chest1_to_seq sy'))
