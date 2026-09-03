module Kuiper.KB.SoftmaxAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module RS = Kuiper.Kernel.RowSoftmax
module Copy = Kuiper.KB.Tensor.Copy

inline_for_extraction noextract
fn softmax_alloc
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
      on gpu_loc (output |-> so) ** pure (so %~ RS.row_softmax_real (reveal r))
{
  let output = Copy.copy_alloc #t (rows *^ cols) input;
  RS.row_softmax_gpu #t rows cols 1024sz output (reveal r);
  return output
}

let softmax_alloc_f32 = softmax_alloc #f32
let softmax_alloc_f64 = softmax_alloc #f64
