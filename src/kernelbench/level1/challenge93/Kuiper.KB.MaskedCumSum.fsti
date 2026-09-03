module Kuiper.KB.MaskedCumSum

(* KernelBench L1 #93: masked cumulative sum along the inner dimension.

   The ABI receives the contiguous storage of a torch.bool mask as bytes.  A
   zero byte gates the corresponding input to real zero and any nonzero byte
   retains it.  The verified implementation performs that gate into a
   verified GPU scratch allocation, scans the gated rows, and frees the
   scratch.  The entry point also allocates its result.  The bridge performs
   only ABI checks and one call to this entry point. *)

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq

let real_mask_step (x : real) (m : u8) : real =
  if m = 0uy then 0.0R else x

let real_masked_row
  (#b #d : nat)
  (sx : chest2 f32 b d)
  (sm : chest2 u8 b d)
  (r : natlt b)
  : GTot (lseq real d)
  = Seq.init_ghost d (fun i ->
      real_mask_step
        (to_real (acc2 sx r i))
        (acc2 sm r i))

let masked_cumsum_post
  (b d : nat)
  (sx : chest2 f32 b d)
  (sm : chest2 u8 b d)
  (sy : chest2 f32 b d)
  : prop
  = forall (r : nat) (i : nat).
      r < b /\ i < d ==>
      acc2 sy r i %~
        rsum (Seq.slice (real_masked_row sx sm r) 0 (i + 1))

fn masked_cumsum_fw_f32
  (b : szp { b <= max_blocks })
  (d : szp {
    SZ.fits (SZ.v b * SZ.v d) /\
    SZ.v b * SZ.v d <= max_blocks * max_threads })
  (input : array2 f32 (l2_row_major b d) { is_global input })
  (mask  : array2 u8  (l2_row_major b d) { is_global mask })
  (#sx : chest2 f32 b d)
  (#sm : chest2 u8 b d)
  preserves
    cpu **
    on gpu_loc (input |-> sx) **
    on gpu_loc (mask |-> sm)
  returns output : array2 f32 (l2_row_major b d)
  ensures
    exists* (sy : chest2 f32 b d).
      on gpu_loc (output |-> sy) **
      pure (masked_cumsum_post b d sx sm sy)

inline_for_extraction let () = ()
