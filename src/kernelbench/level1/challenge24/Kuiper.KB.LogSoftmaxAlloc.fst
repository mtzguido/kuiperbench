module Kuiper.KB.LogSoftmaxAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module RLS = Kuiper.Kernel.RowLogSoftmax
module Copy = Kuiper.KB.Tensor.Copy

inline_for_extraction noextract
fn logsoftmax_alloc
  (#t:Type0) {| floating t, real_like t, floating_real_like t |}
  (rows : szp { 0 < rows /\ rows <= max_blocks })
  (cols : szp { 0 < cols /\ rows * cols <= max_blocks * max_threads })
  (input : array2 t (l2_row_major rows cols) { is_global input })
  (#s : chest2 t rows cols)
  (r : erased (chest2 real rows cols))
  (#f : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (input |-> Frac f s) ** pure (s %~ reveal r)
  returns output : array2 t (l2_row_major rows cols)
  ensures
    exists* (so : chest2 t rows cols).
      on gpu_loc (output |-> so) **
      pure (so %~ RLS.row_log_softmax_real (reveal r))
{
  let output = Copy.copy_alloc #t (rows *^ cols) input;
  RLS.row_log_softmax_gpu #t rows cols output (reveal r);
  return output
}

let logsoftmax_alloc_f32 = logsoftmax_alloc #f32
let logsoftmax_alloc_f64 = logsoftmax_alloc #f64
