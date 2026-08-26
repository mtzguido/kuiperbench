module Kuiper.KB.MinReduceDim

(* KernelBench L1 #53: min reduction over the middle dimension of a
   (B, D, M) row-major tensor.  PyTorch reference:
       y = torch.min(x, dim=1).values      # shape (B, M)

   Kuiper view: factor the (B, D, M) buffer as a 2-D matrix of shape
   (B*M, D) using [Kuiper.Tensor.Layout.BCMPages.l2_bcm_pages] with
   parameters (b=B, hw=M, c=D).  Row r = b*M + j carries the length-D
   slice x[b, :, j].  A single launch of the one-thread-per-row
   primitive [Kuiper.Kernel.HReduce.Min.reduce_batched_min_f32]
   produces, per row r, the f32 min of that row (i.e., y[b, j]).

   Unlike sum-reduction, IEEE-754 [fmin] on the modeled carrier is
   bit-exactly associative+commutative, so the post is exact equality
   (no [%~]).

   The output is a flat length-(B*M) Array1; the bridge reshapes it
   to (B, M) on the host for the harness comparison.

   Exactly 1 GPU kernel launch.  No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.MinReduceDim
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq

val minreduce_dim_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads })
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v m) (SZ.v d)) { is_global x })
  (y : array1 f32 (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : erased (EM.chest2 f32 (SZ.v b * SZ.v m) (SZ.v d)))
  (#sy : erased (chest1 f32 (SZ.v b * SZ.v m)))
  : stt unit
      (cpu **
       on gpu_loc (x |-> sx) **
       on gpu_loc (y |-> sy))
      (fun _ ->
        cpu **
        on gpu_loc (x |-> sx) **
        (exists* (sy' : chest1 f32 (SZ.v b * SZ.v m)).
           on gpu_loc (y |-> sy') **
           pure (minreduce_post (SZ.v b * SZ.v m) (SZ.v d) sx (chest1_to_seq sy'))))
