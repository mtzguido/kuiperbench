module Kuiper.KB.BatchedGEMM

(* KernelBench L1 #3 — Batched matrix multiplication using Array3.
   Inputs are array3 on GPU with l3_batched_row_major layout.
   Per-batch kernel launch using the verified Naive2 GEMM.
   Layouts instantiated to row-major (no function pointers in extracted code).
   Zero assume · zero magic · zero admit.

   Functional spec: the output is page-wise the matmul of the input pages,
   i.e. out i j k == matmul (slice_page sa i) (slice_page sb i) j k. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

(* Per-page batched matmul spec.  Identical to [Kuiper.Spec.GEMM.batched_matmul]:
   page [i] of the output is the matmul of page [i] of each input. *)
let batched_matmul
  (#et:Type) {| scalar et |}
  (#batch #rows #shared #cols : nat)
  (a : chest3 et batch rows shared)
  (b : chest3 et batch shared cols)
  : chest3 et batch rows cols
  = mk3 fun i j k ->
      acc2 (MS.matmul (slice_page a i)
                      (slice_page b i)) j k

(* The caller provides the output array [c]; the kernel overwrites every page
   (the per-page GEMM uses [comb2], which ignores prior content), so the
   initial contents [sc0] are arbitrary. This avoids an internal allocation
   plus a device-to-device copy in the bridge. *)
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
