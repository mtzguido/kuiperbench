module Kuiper.KB.GemmDivSumScale

(* KernelBench L2 #14: Gemm_Divide_Sum_Scaling.

   PyTorch reference (no bias):
     x  = matmul(x, W.T)          # (batch, hidden)
     x  = x / 2                   # divide
     x  = sum(x, dim=1, keepdim)  # (batch, 1)  -- sum over hidden
     x  = x * scaling_factor      # scale
   Output shape (batch, 1), i.e. one scalar per row.

   Algebra exploited:
       sum(matmul/2) * s == (sum over hidden of matmul) * (s/2)
   so the two scalars fold into a single constant k = scaling_factor/2.

   Pipeline (3 GPU launches), caller passes Wᵀ as [wt] (input × hidden):
     1. GEMM:  gC := x @ wt           (Kuiper.Kernel.GEMM.Naive2.mmcomb_gpu_approx)
     2. Row-sum over hidden into y    (Kuiper.Kernel.HReduce.Block.reduce_batched_block)
     3. Scalar-multiply y by k        (Kuiper.Kernel.Map.map_gpu / Kuiper.KB.ScalarMul)

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Per-row functional postcondition: each output element approximates the
   row-sum of the real-valued matmul (x @ wt), scaled by the real of [k]. *)
let gdss_post
  (#t:Type0) {| scalar t, real_like t |}
  (#batch #input #hidden : nat)
  (k : t)
  (sx  : chest2 t batch input)
  (swt : chest2 t input hidden)
  (sy' : chest1 t batch)
  : prop
  = forall (r:nat). r < batch ==>
      (acc1 sy' r) %~
        (rsum (EM.ematrix_row
                 (MS.matmul (EM.to_real_matrix sx) (EM.to_real_matrix swt)) r)
         *. to_real k)

fn gemm_div_sum_scale_f32
  (batch input : szp)
  (hidden : szp {
     SZ.v batch <= max_blocks /\
     SZ.v batch * SZ.v hidden <= max_blocks * max_threads /\
     SZ.fits (SZ.v hidden + max_threads) /\
     SZ.fits (SZ.v batch * SZ.v input) /\
     SZ.fits (SZ.v input * SZ.v hidden) /\
     SZ.fits (SZ.v batch * SZ.v hidden) })
  (k : f32)
  (x  : array2 f32 (l2_row_major batch input)  { is_global x  })
  (wt : array2 f32 (l2_row_major input hidden) { is_global wt })
  (y  : array1 f32 (l1_forward batch)                 { is_global y  })
  (#sx  : chest2 f32 batch input)
  (#swt : chest2 f32 input hidden)
  (#sy  : chest1 f32 batch)
  preserves
    cpu **
    on gpu_loc (x  |-> sx) **
    on gpu_loc (wt |-> swt)
  requires
    on gpu_loc (y  |-> sy)
  ensures
    (exists* (sy' : chest1 f32 batch).
       on gpu_loc (y |-> sy') **
       pure (gdss_post k sx swt sy'))
