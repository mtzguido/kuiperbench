module Kuiper.KB.ArgminReduceDim

(* KernelBench L1 #52: argmin over the middle dimension of a (B, D, M)
   row-major tensor.  PyTorch reference:
       y = torch.argmin(x, dim=1)         # shape (B, M), dtype int64

   Kuiper view: factor (B, D, M) as a 2-D matrix of shape (B*M, D)
   using Kuiper.Tensor.Layout.BCMPages.l2_bcm_pages.  Row r = b*M + j
   carries the length-D slice x[b, :, j].  One launch of
   Kuiper.Kernel.HReduce.Argmin.reduce_batched_argmin_f32 produces, per
   row r, an i64 index bi such that
       0 <= bi < D
       x[b, bi, j] == min_k x[b, k, j]
       for all k' < bi.  x[b, k', j] =/= min_k x[b, k, j].

   Tie-break: the kernel's strict-less-than update implements the
   PyTorch "first occurrence" convention, and we now *verify* it: the
   returned index is the first (smallest) column attaining the row
   minimum.  See skeptic.txt.

   Exactly 1 GPU kernel launch.  No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Math.Fmin
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMPages
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq
module I64 = FStar.Int64

(* Per-row post: the i64 output is in-bounds and is the *first*
   (smallest-index) column attaining the row minimum — the PyTorch
   first-occurrence argmin tie-break. *)
let argminreduce_post
  (n_rows : nat) (n_cols : nat{n_cols > 0})
  (sx : EM.chest2 f32 n_rows n_cols)
  (sy : Seq.lseq I64.t n_rows)
  : prop =
  forall (r : nat). r < n_rows ==>
    (let bi = I64.v (Seq.index sy r) in
     0 <= bi /\ bi < n_cols /\
     acc2 sx r bi == seq_fmin (EM.ematrix_row sx r) /\
     (forall (j : nat). j < bi ==>
        ~(acc2 sx r j == seq_fmin (EM.ematrix_row sx r))))

fn argminreduce_dim_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads /\
             SZ.v d < pow2 63 })
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v m) (SZ.v d)) { is_global x })
  (y : array1 I64.t (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : erased (EM.chest2 f32 (SZ.v b * SZ.v m) (SZ.v d)))
  (#sy : erased (chest1 I64.t (SZ.v b * SZ.v m)))
  preserves cpu ** on gpu_loc (x |-> sx)
  requires on gpu_loc (y |-> sy)
  ensures
    exists* (sy' : chest1 I64.t (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy') **
      pure (argminreduce_post (SZ.v b * SZ.v m) (SZ.v d) sx (chest1_to_seq sy'))
