module Kuiper.KB.L2Norm

(* KernelBench L1 #39: row-wise L2 normalisation along dim=1.
   Input X is a (B, D) row-major tensor, viewed flat.
   For each row r:    X[r,:] ← X[r,:] / sqrt(sum_j X[r,j]^2)
   In place; KernelBench does not require X to be preserved.

   Composes verified Kuiper primitives:
     - Kuiper.Kernel.HReduce.reduce  (sum-of-squares with sq_step pre-map)
     - Kuiper.Kernel.Map.map_gpu     (in-place pointwise scale)
   Per row uses the device-to-device offset memcpy primitive to copy a
   row in/out of a fixed-size scratch buffer.

   Functional postcondition (see [Kuiper.Spec.L2Norm]).  Each output
   row [r] is a uniform scaling [frobenius_result inv_r] of the
   corresponding input row, where [inv_r == rsqrt sumsq_r] and
   [sumsq_r] approximates the row's real-valued sum-of-squares.  Both
   [inv_r] and [sumsq_r] are existentially bound per row because the
   device-side reduction only approximates the real sum and [rsqrt]
   is opaque to the spec.

   Edge case (all-zero row): see Kuiper.Spec.L2Norm.  No precondition
   required; matches the PyTorch reference (NaNs out). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.L2Norm
module SZ = Kuiper.SizeT

inline_for_extraction noextract
type l2norm_fw_ty (t:Type0) {| scalar t, real_like t, floating t |} =
  (b : szp) ->
  (d : szp { d <= max_blocks * max_threads /\
             SZ.fits (b * d) /\
             b * d <= max_blocks * max_threads }) ->
  (x : array1 t (l1_forward (b *^ d)) { is_global x }) ->
  (#s : erased (chest1 t (b *^ d))) ->
  stt unit
    (requires cpu ** on gpu_loc (x |-> s))
    (ensures fun _ ->
      cpu **
      (exists* (s' : chest1 t (b *^ d)).
         on gpu_loc (x |-> s') **
         pure (l2norm_post (SZ.v b) (SZ.v d) (chest1_to_seq s) (chest1_to_seq s'))))

val l2norm_fw_f32 : l2norm_fw_ty f32
