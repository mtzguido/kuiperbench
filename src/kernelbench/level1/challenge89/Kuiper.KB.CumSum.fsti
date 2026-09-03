module Kuiper.KB.CumSum

(* KernelBench L1 #89: cumulative sum (prefix sum) along the inner
   dimension of a 2-D (B, D) row-major tensor.  PyTorch reference:
       y = torch.cumsum(x, dim=1)                    # shape (B, D)

   Kuiper view: an [array2 f32] of shape [b * d] under
   [l2_row_major].  Each row is a length-[d] f32 sequence; the output
   row at column [i] is the f32-fold of the input row [0..i+1].

   The per-row work is delegated to
   [Kuiper.Kernel.Scan1D.RowBlock.scan1d_inclusive_rowblock], which
   runs one block per row (single-threaded) doing the sequential
   inclusive scan.  Size bound: [b <= max_blocks].

   The postcondition lifts the bit-exact f32 fold to an
   [%~]-approximation of the real-arithmetic ideal cumulative sum
   ([rsum] over the corresponding slice of [to_real_seq] of the row),
   following the [Kuiper.Spec.SumReduceDim] / [Kuiper.Spec.Frobenius]
   convention.

   No assume / magic / admit.  Exactly 1 GPU kernel launch. *)

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq

(* Per-cell post: cell [(r, i)] is %~-equal to the real-arithmetic
   inclusive prefix sum of the first [i+1] real-lifted inputs of row [r]. *)
let cumsum_post
  (#t : Type0) {| scalar t, real_like t |}
  (b d : nat)
  (sx : chest2 t b d)
  (sy : chest2 t b d)
  : prop
  = forall (r : nat) (i : nat).
      r < b /\ i < d ==>
      acc2 sy r i %~
        rsum (Seq.slice (to_real_seq (EM.ematrix_row sx r)) 0 (i + 1))

fn cumsum_fw_f32
  (b : szp { b <= max_blocks })
  (d : szp { SZ.fits (SZ.v b * SZ.v d) })
  (input  : array2 f32 (l2_row_major b d)
            { is_global input  })
  (output : array2 f32 (l2_row_major b d)
            { is_global output })
  (#sx #sy0 : chest2 f32 b d)
  preserves
    cpu **
    on gpu_loc (input  |-> sx)
  requires
    on gpu_loc (output |-> sy0)
  ensures
    (exists* (sy : chest2 f32 b d).
       on gpu_loc (output |-> sy) **
       pure (cumsum_post b d sx sy))

fn cumsum_alloc_f32
  (b : szp { b <= max_blocks })
  (d : szp { SZ.fits (SZ.v b * SZ.v d) })
  (input : array2 f32 (l2_row_major b d) { is_global input })
  (#sx : chest2 f32 b d)
  preserves cpu ** on gpu_loc (input |-> sx)
  returns output : array2 f32 (l2_row_major b d)
  ensures
    exists* (sy : chest2 f32 b d).
      on gpu_loc (output |-> sy) ** pure (cumsum_post b d sx sy)


inline_for_extraction let () = ()
