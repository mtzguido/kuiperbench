module Kuiper.Kernel.RowLogSoftmax

(* Per-row LogSoftmax over an [Array2 et m n] resident on the GPU.

   Implementation: TWO kernel launches (independent of m).
     1. [Kuiper.Kernel.HReduce.Block.reduce_batched_block exp rexp]
        — one block per row, tree-reduces row sums of exp(x).
     2. [Kuiper.Kernel.RowSubLog.row_sub_log]
        — one thread per cell (i, j); writes
          [a[i, j] := a_old[i, j] - log(sums[i])] in place.

   This is the row-batched analog of [Kuiper.Kernel.LogSoftmax]
   (which operates on a single row), built on the
   [Kuiper.Kernel.RowBroadcast] + [HReduce.Block] primitives. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor
(* All-real specification: each row independently log_softmax'd. *)
let row_log_softmax_real
  (#m : nat) (#n : nat { n > 0 })
  (ra : chest2 real m n)
  : chest2 real m n
  = mk2 (fun i j ->
      acc1 (Kuiper.Kernel.LogSoftmax.log_softmax_real (chest2_row ra i)) j)

inline_for_extraction noextract
type row_log_softmax_gpu_ty
  (et : Type0) {| floating et, real_like et, floating_real_like et |} =
  fn (m : szp { m <= max_blocks })
     (n : szp { m * n <= max_blocks * max_threads })
  (#l : layout2 m n) {| ctlayout l |}
  (a : array2 et l { is_global a })
  (#sa : erased (chest2 et m n))
  (ra : erased (chest2 real m n))
  preserves cpu
  requires
    on gpu_loc (a |-> sa) **
    pure (sa %~ ra)
  ensures
    exists* (sa' : chest2 et m n).
      on gpu_loc (a |-> sa') **
      pure (sa' %~ row_log_softmax_real ra)

inline_for_extraction noextract
val row_log_softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : row_log_softmax_gpu_ty et
