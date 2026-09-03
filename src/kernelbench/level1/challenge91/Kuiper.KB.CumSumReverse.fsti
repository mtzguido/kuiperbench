module Kuiper.KB.CumSumReverse

(* KernelBench L1 #91: reverse cumulative sum along the inner dimension.

   The source operation is

     reverse (cumsum (reverse input))

   for every row of a two-dimensional f32 tensor.  The implementation uses a
   zero-copy, verified reversal of the input and output tensor layouts around
   the verified row-block scan.  The entry point also allocates its result.
   Consequently the bridge performs no flip, transpose, allocation, or
   arithmetic: it checks the ABI and calls this entry point.

   The postcondition is direct: output cell [(r,i)] approximates the real sum
   of the first [d-i] elements of the reversed real input row, in exactly the
   order induced by the PyTorch expression above. *)

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Spec.Scan1D { seq_rev }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq

let cumsum_reverse_post
  (b d : nat)
  (sx : chest2 f32 b d)
  (sy : chest2 f32 b d)
  : prop
  = forall (r : nat) (i : nat).
      r < b /\ i < d ==>
      acc2 sy r i %~
        rsum
          (Seq.slice
            (seq_rev (to_real_seq (EM.ematrix_row sx r)))
            0 (d - i))

fn cumsum_reverse_fw_f32
  (b : szp { b <= max_blocks })
  (d : szp { SZ.fits (SZ.v b * SZ.v d) })
  (input : array2 f32 (l2_row_major b d) { is_global input })
  (#sx : chest2 f32 b d)
  preserves
    cpu **
    on gpu_loc (input |-> sx)
  returns output : array2 f32 (l2_row_major b d)
  ensures
    exists* (sy : chest2 f32 b d).
      on gpu_loc (output |-> sy) **
      pure (cumsum_reverse_post b d sx sy)

inline_for_extraction let () = ()
