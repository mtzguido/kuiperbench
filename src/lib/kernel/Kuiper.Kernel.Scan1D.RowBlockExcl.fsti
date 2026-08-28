module Kuiper.Kernel.Scan1D.RowBlockExcl

(* Row-per-block sequential *exclusive* prefix-scan primitive.
 *
 * Sibling of [Kuiper.Kernel.Scan1D.RowBlock] (which is inclusive).
 * Same execution structure: ONE BLOCK PER ROW, single thread per
 * block, doing the full sequential row scan inside the single
 * thread.  Each row is independent -> no barriers, no shared
 * memory, no carry propagation.  Total work O(rows * cols),
 * parallelized across rows.
 *
 * The only difference from the inclusive variant is the per-cell
 * postcondition: cell [(r, j)] holds the *exclusive* prefix fold
 * [scan_exclusive_at m row_r j == red_fold m m.rid (slice row_r 0 j)],
 * i.e. the fold over the strictly-earlier elements [row_r[0 .. j)].
 * Cell 0 is therefore the reducer seed [m.rid].  This is achieved
 * by writing the running accumulator into the output cell *before*
 * folding the cell's own input element into it.
 *
 * The only size bound is [rows <= max_blocks].
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

(* The result chest2: cell [(r, j)] is [scan_exclusive_at m row_r j]. *)
let scan2d_exclusive_result
  (#t : Type0)
  (m : reducer t)
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  : chest2 t rows cols
  = mk2 (fun (r : natlt rows) (j : natlt cols) ->
      scan_exclusive_at m (EM.ematrix_row sx r) j)

unfold inline_for_extraction
type scan1d_exclusive_rowblock_ty =
  fn (#et : Type0) {| scalar et |}
     (m : reducer et)
     (rows : szp { rows <= max_blocks })
     (cols : szp)
     (#lin  : layout2 rows cols) {| ctlayout lin  |}
     (#lout : layout2 rows cols) {| ctlayout lout |}
     (input  : array2 et lin  { is_global input  })
     (output : array2 et lout { is_global output })
     (#sx #sout : chest2 et rows cols)
     (#fIn  : perm)
     preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
     requires
       on gpu_loc (output |-> sout)
     ensures
       on gpu_loc (output |-> scan2d_exclusive_result m sx)

inline_for_extraction noextract
val scan1d_exclusive_rowblock : scan1d_exclusive_rowblock_ty
