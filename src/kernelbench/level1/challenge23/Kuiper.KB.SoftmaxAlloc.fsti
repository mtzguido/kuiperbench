module Kuiper.KB.SoftmaxAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Kernel.RowSoftmax { row_softmax_real }

inline_for_extraction noextract
type softmax_alloc_ty
  (t:Type0) {| floating t, real_like t, floating_real_like t |} =
  fn (rows : szp { 0 < rows /\ rows <= max_blocks })
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
         on gpu_loc (output |-> so) ** pure (so %~ row_softmax_real (reveal r))

val softmax_alloc_f32 : softmax_alloc_ty f32
val softmax_alloc_f64 : softmax_alloc_ty f64
