module Kuiper.Kernel.Scan1D.RowBlock

(* Row-per-block sequential inclusive prefix-scan primitive.
 *
 * Sibling of [Kuiper.Kernel.Scan1D].  The existing primitive uses
 * one CUDA thread per output cell (cell-parallel) and runs an O(j)
 * sequential fold per cell, for total work O(rows * cols^2).  For
 * the KernelBench L1 scan inputs (32768 × 32768) that's 2^45
 * float-ops — infeasible.
 *
 * This primitive uses ONE BLOCK PER ROW, single thread per block,
 * doing the full sequential row scan inside the single thread.
 * Each row is independent → no barriers, no shared memory, no
 * carry propagation.  Total work O(rows * cols), parallelized
 * across rows.  For 32768x32768 that's ~32768 blocks of 32768
 * sequential ops each running on SMs in parallel.
 *
 * The only size bound is [rows <= max_blocks].  [cols] is
 * unconstrained beyond what [layout2]'s own [SZ.fits]
 * obligation already requires.
 *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Scan1D
open Kuiper.Monoid.Reduce
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module Seq = FStar.Seq

(* The result chest2: cell [(r, j)] is [scan_inclusive_at m row_r j]. *)
let scan2d_inclusive_result
  (#t : Type0)
  (m : cmonoid t)
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  : EM.chest2 t rows cols
  = mk2 (fun (r : natlt rows) (j : natlt cols) ->
      scan_inclusive_at m (EM.ematrix_row sx r) j)

unfold inline_for_extraction
type scan1d_inclusive_rowblock_ty =
  fn (#et : Type0) {| scalar et |}
     (m : cmonoid et)
     (rows : szp { rows <= max_blocks })
     (cols : szp)
     (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
     (#lout : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lout |}
     (input  : array2 et lin  { is_global input  })
     (output : array2 et lout { is_global output })
     (#sx #sout : erased (EM.chest2 et (SZ.v rows) (SZ.v cols)))
     (#fIn  : perm)
     preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
     requires
       on gpu_loc (output |-> sout)
     ensures
       on gpu_loc (output |-> scan2d_inclusive_result m sx)

inline_for_extraction noextract
val scan1d_inclusive_rowblock : scan1d_inclusive_rowblock_ty
