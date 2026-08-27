module Kuiper.KB.CumSumExclusive

(* KernelBench L1 #92: *exclusive* cumulative sum (prefix sum) along
   the inner dimension of a 2-D (B, D) row-major tensor.  PyTorch
   reference:
       cumsum = torch.cumsum(x[:, :-1], dim=1)
       y = cat(zeros[:, 0:1], cumsum, dim=1)         # shape (B, D)

   Equivalently: [y[b, j] = sum_{i < j} x[b, i]], with [y[b, 0] = 0].
   This is exactly the exclusive prefix scan of each row.

   The per-row work is delegated to
   [Kuiper.Kernel.Scan1D.RowBlockExcl.scan1d_exclusive_rowblock],
   which runs one block per row (single-threaded) doing the
   sequential *exclusive* scan: it writes the running accumulator
   into each output cell BEFORE folding the cell's own input element,
   so cell [(b, j)] holds the fold of the strictly-earlier elements
   [x[b, 0 .. j)].  No host-side subtraction or shift is performed —
   the exclusive transform lives entirely inside the verified kernel.

   The postcondition lifts the bit-exact f32 fold to an
   [%~]-approximation of the real-arithmetic ideal exclusive
   cumulative sum ([rsum] over the [0, i) slice of [to_real_seq] of
   the row).

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
   *exclusive* prefix sum of the first [i] real-lifted inputs of row
   [r] (i.e. over the strictly-earlier elements).  Cell 0 is the sum
   over the empty slice, [0.0]. *)
let cumsum_exclusive_post
  (#t : Type0) {| scalar t, real_like t |}
  (b d : nat)
  (sx : EM.chest2 t b d)
  (sy : EM.chest2 t b d)
  : prop
  = forall (r : nat) (i : nat).
      r < b /\ i < d ==>
      acc2 sy r i %~
        rsum (Seq.slice (to_real_seq (EM.ematrix_row sx r)) 0 i)

inline_for_extraction noextract
type cumsum_exclusive_fw_ty (t:Type0) {| scalar t, real_like t |} =
  fn (b : szp { b <= max_blocks })
     (d : szp { SZ.fits (SZ.v b * SZ.v d) })
     (input  : array2 t (l2_row_major (SZ.v b) (SZ.v d))
               { is_global input  })
     (output : array2 t (l2_row_major (SZ.v b) (SZ.v d))
               { is_global output })
     (#sx #sy0 : EM.chest2 t (SZ.v b) (SZ.v d))
     requires
       cpu **
       on gpu_loc (input  |-> sx) **
       on gpu_loc (output |-> sy0)
     ensures
       cpu **
       on gpu_loc (input |-> sx) **
       (exists* (sy : EM.chest2 t (SZ.v b) (SZ.v d)).
          on gpu_loc (output |-> sy) **
          pure (cumsum_exclusive_post (SZ.v b) (SZ.v d) sx sy))

val cumsum_exclusive_fw_f32 : cumsum_exclusive_fw_ty f32

inline_for_extraction let () = ()
