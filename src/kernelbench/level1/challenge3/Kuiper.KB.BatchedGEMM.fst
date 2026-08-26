module Kuiper.KB.BatchedGEMM

(* KernelBench L1 #3 — Batched matrix multiplication using rank-3 tensors.
   Inputs are array3 on GPU with l3_batched_row_major layout.  This is a thin
   orchestrator over the verified batched GEMM kernel
   [Kuiper.Kernel.BatchedGEMM.bmmcomb_gpu_exact], instantiated with [comb2]
   (which ignores prior output content, so the per-page result is exactly the
   matmul of the input pages).
   Zero assume · zero magic · zero admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module BG = Kuiper.Kernel.GEMM.Naive2

#push-options "--z3rlimit 40"
fn batched_gemm_f32
  (batch rows shared cols : szp)
  (a : array3 f32 (l3_batched_row_major batch rows shared) { is_global a })
  (b : array3 f32 (l3_batched_row_major batch shared cols) { is_global b })
  (c : array3 f32 (l3_batched_row_major batch rows cols) { is_global c })
  (#sa : erased (chest3 f32 batch rows shared))
  (#sb : erased (chest3 f32 batch shared cols))
  (#sc0 : erased (chest3 f32 batch rows cols))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> Frac fA sa) **
    on gpu_loc (b |-> Frac fB sb)
  requires
    on gpu_loc (c |-> sc0) **
    pure (
      batch * (rows * cols) <= max_blocks * max_threads /\
      SZ.fits (batch * (rows * shared)) /\
      SZ.fits (batch * (shared * cols)) /\
      SZ.fits (batch * (rows * cols))
    )
  ensures
    on gpu_loc (c |-> batched_matmul sa sb)
{
  BG.bmmcomb_gpu_exact #f32 MS.comb2 batch rows cols shared
    #(l3_batched_row_major _ _ _)
    #(l3_batched_row_major _ _ _)
    #(l3_batched_row_major _ _ _)
    a b c;
  with sc'. assert on gpu_loc (c |-> sc');
  assert pure (equal sc' (batched_matmul sa sb));
  ()
}
#pop-options
