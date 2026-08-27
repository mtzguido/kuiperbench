module Kuiper.KB.L1Norm

(* KernelBench L1 #38: row-wise L1 normalisation along dim=1.
   Reference: y = x / mean(|x|, dim=1, keepdim=True)
            = x * D / sum(|x|, dim=1, keepdim=True)

   Input X is a (B, D) row-major tensor, viewed in F* as an Array2
   with the plain contiguous layout [l2_row_major B D] (so the matrix
   entry [acc2 sx r j] is the physical element X[r,j]).  The kernel
   normalises in place; KernelBench does not require X to be preserved.

   Exactly 3 GPU kernel launches over the matrix:
     1. reduce_batched (x ↦ |x|)   -- sum of absolutes per row
     2. map_gpu (s ↦ D / s)         -- per-row scale factor
     3. row_scale                   -- in-place scale each row

   The scaling factor [D / s] needs the float-valued dimension [D];
   it is built inside the verification boundary as [l1_dim_f d =
   of_int (int64 D)] (extracts to (float)(int64_t)(uint64_t)D), so no
   unverified floating-point arithmetic happens in the C bridge.

   Functional postcondition (see [Kuiper.Spec.L1Norm]): each output
   row [r] is a uniform scaling [mul scale_r] of the corresponding
   input row, where [scale_r == div (l1_dim_f d) sum_abs_r] and
   [sum_abs_r] approximates the row's real-valued L1 norm.  Both are
   existentially bound per row because the device-side reduction only
   approximates the real sum and [div] is opaque to the spec.

   Edge case (all-zero row): see Kuiper.Spec.L1Norm.  Matches the
   PyTorch reference (NaNs out).

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Spec.L1Norm
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT

(* The float-valued row dimension D, built from the runtime [d] inside
   the verification boundary (extracts to (float)(int64_t)(uint64_t)D)
   so no unverified floating-point arithmetic happens in the C bridge. *)
inline_for_extraction noextract
let l1_dim_f (#t:Type0) {| floating t |} (d : SZ.t) : t =
  of_int (FStar.Int.Cast.uint64_to_int64
            (FStar.SizeT.sizet_to_uint64 d))

inline_for_extraction noextract
type l1norm_fw_ty (t:Type0) {| scalar t, real_like t, floating t |} =
  fn (b : szp)
     (d : szp { 0 < SZ.v d /\ SZ.fits (SZ.v b * SZ.v d) })
     (x : array2 t (l2_row_major b d) { is_global x })
     (#sx : EM.chest2 t b d)
     requires
       cpu **
       on gpu_loc (x |-> sx) **
       pure (
         SZ.v b > 0 /\
         SZ.v b * SZ.v d <= max_blocks * max_threads
       )
     ensures
       cpu **
       (exists* (sx' : EM.chest2 t b d).
          on gpu_loc (x |-> sx') **
          pure (l1norm_post b d (l1_dim_f d) sx sx'))

val l1norm_fw_f32 : l1norm_fw_ty f32
