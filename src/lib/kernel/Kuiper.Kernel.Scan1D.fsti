module Kuiper.Kernel.Scan1D

(* Polymorphic 2-D row-batched inclusive prefix-scan primitive.
 *
 * Input:  (rows × cols) chest2 of [t]
 * Output: (rows × cols) chest2 of [t]
 *         where row [r] is [scan_inclusive m (ematrix_row sx r)],
 *         i.e. cell [(r, j)] = [m.rop]-fold over input[r, 0..j+1].
 *
 * The implementation uses one CUDA thread per output cell
 * (gid = r * cols + j); each thread reads input[r, 0..j+1]
 * sequentially and folds with [m.rop].  Total work is therefore
 * O(rows * cols^2) — the simplest scan layout that does not
 * require shared-memory barriers or carry-propagation between
 * blocks.  The two consequences:
 *
 *   - Cell-parallel: rows * cols ≤ max_blocks * max_threads.
 *   - Per-cell sequential read: cols may be arbitrary up to the
 *     usual SZ.fits / max_blocks * max_threads bound, but the
 *     wall-clock cost per cell is O(j).
 *
 * For the KernelBench L1 scan inputs (32768 × 32768) the cell
 * count exceeds [max_blocks * max_threads]; the primitive therefore
 * does *not* directly cover those inputs (a multi-block Hillis-
 * Steele or Blelloch scan would, see STATUS).  It does cover any
 * (rows × cols) with rows * cols ≤ max_blocks * max_threads, which
 * subsumes scan dims ≤ ~1024 across plausible row counts and
 * suffices to verify the spec / wrapper plumbing.
 *
 * The .fst follows the one-thread-per-output-cell layout of
 * [Kuiper.Kernel.WindowReduce1D].  The implementation is fully
 * verified with no admits/assumes. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Scan1D
open Kuiper.Monoid.Reduce
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module Seq = FStar.Seq

(* ── Spec helper: the result chest2 ──────────────────────────────────── *)

let scan2d_inclusive_result
  (#t : Type0)
  (m : cmonoid t)
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  : EM.chest2 t rows cols
  = mk2 (fun (r : natlt rows) (j : natlt cols) ->
      scan_inclusive_at m (EM.ematrix_row sx r) j)

(* ── Polymorphic core ────────────────────────────────────────────────── *)

unfold inline_for_extraction
type scan1d_inclusive_ty =
  fn (#et : Type0) {| scalar et |}
     (m : cmonoid et)
     (rows : szp)
     (cols : szp)
     (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
     (#lout : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lout |}
     (input  : array2 et lin  { is_global input  })
     (output : array2 et lout { is_global output })
     (#sx   : erased (EM.chest2 et (SZ.v rows) (SZ.v cols)))
     (#sout : erased (EM.chest2 et (SZ.v rows) (SZ.v cols)))
     (#fIn  : perm)
     preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
     requires
       on gpu_loc (output |-> sout) **
       pure (SZ.v rows * SZ.v cols <= max_blocks * max_threads) **
       pure (SZ.fits (SZ.v rows * SZ.v cols))
     ensures
       on gpu_loc (output |-> scan2d_inclusive_result m sx)

inline_for_extraction noextract
val scan1d_inclusive : scan1d_inclusive_ty
