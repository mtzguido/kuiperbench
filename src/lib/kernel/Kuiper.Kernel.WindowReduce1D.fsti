module Kuiper.Kernel.WindowReduce1D

(* ────────────────────────────────────────────────────────────────────────
 * WIP CONTRACT-ONLY .fsti — *not* on main.
 *
 * This file is the intended interface for the [WindowReduce1D] Pulse
 * primitive that backs KernelBench L1 #41-#46 (1-D / 2-D / 3-D max-pool
 * and avg-pool).  It is checked in on the [kb-windowreduce1d-wip] feature
 * branch as a handoff artifact for the next session; there is no .fst yet
 * and the kernel itself is *not* axiomatised on main.  See
 * [src/kernelbench/level1/POOLING_DESIGN.md] §2 and §5 for the design
 * rationale and resume instructions.
 *
 * Two monomorphic instantiations are intended on top of the polymorphic
 * core, mirroring the [maxpool1d_post] / [avgpool1d_post] postconditions
 * exposed by [Kuiper.Spec.Pool1D]:
 *
 *   - [windowreduce_max_f32] : reduces by [fmax] (exact equality post,
 *     using the [Kuiper.Math.Fmax] axioms or the [cmonoid_fmax_f32]
 *     instance from [Kuiper.Monoid.Reduce.F32]).
 *   - [windowreduce_plus_f32] : reduces by [+] (approximate [%~] post).
 *
 * The polymorphic core takes a [cmonoid t] explicitly (not a
 * [tcinstance]) since two reduction monoids on the same carrier would
 * be ambiguous to type-class resolution.
 *
 * ────────────────────────────────────────────────────────────────────────
 * The core proof obligation is the K-fold *overlapping* permission
 * split: when stride S < kernel K, every input element [i] is read by
 * up to ⌈K/S⌉ output threads, so the input permission must be split
 * fractionally across overlapping windows.  The setup is the long
 * pole of this primitive (~250 LOC of [forevery_factor'] +
 * [forevery_zip] plumbing per the design doc).
 *
 * For the trivial subcase S = K (no overlap, e.g. #45), the setup
 * collapses to the [RowScale]-style pattern.  Implementing the
 * trivial-subcase first is the recommended bring-up path — see
 * design doc §5 step 5. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Pool1D
open Kuiper.Monoid.Reduce
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module Seq = FStar.Seq

(* ── Spec helpers (also defined in the .fst, exposed here for the post) ─ *)

let oob_window
  (#t : Type0) (m : cmonoid t)
  (#l : nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat)
  : GTot (Seq.lseq t k)
  = Seq.init_ghost k (fun di ->
      if pool_in_bounds l s p d j di
      then Seq.index row (pool_input_idx l s p d j di)
      else m.rid)

let window_red
  (#t : Type0) (m : cmonoid t)
  (#l : nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat)
  : GTot t
  = seq_fold_left m.rop m.rid (oob_window m row k s p d j)

let ematrix_to_row (#t : Type0) (#rows #l : nat)
  (sx : EM.chest2 t rows l) (r : natlt rows)
  : GTot (Seq.lseq t l)
  = Seq.init_ghost l (fun c -> acc2 sx r c)

let windowreduce_result
  (#t : Type0) (m : cmonoid t)
  (#rows #l : nat)
  (sx : EM.chest2 t rows l)
  (k s p d lo : nat)
  : EM.chest2 t rows lo
  = mk2 (fun (r : natlt rows) (j : natlt lo) ->
      window_red m (ematrix_to_row sx r) k s p d j)

(* ── Polymorphic core ────────────────────────────────────────────────── *)

(* Per-row windowed reduction over an arbitrary [cmonoid].  The output
 * row is filled with one [m.rop]-fold per output slot, where each
 * fold ranges over the in-bounds elements of a K-wide dilated window
 * starting at output position [j].
 *
 * The functional content of the postcondition is left abstract here:
 * the polymorphic core only guarantees that the output value at slot
 * [j] equals the [m.rop]-fold over the in-bounds taps starting from
 * [m.rid] (or from the first in-bounds element when the caller-supplied
 * "skip-empty" flag is set, for the max-pool [option t] semantics).
 *
 * The two monomorphic wrappers ([windowreduce_max_f32] /
 * [windowreduce_plus_f32]) bridge from this generic post to the
 * [maxpool1d_post] / [avgpool1d_post] specifications. *)
unfold inline_for_extraction
type windowreduce_ty =
  fn (#et : Type0) {| scalar et |}
     (m : cmonoid et)
     (k s : szp)
     (p : sz)
     (d : szp)
     (rows : szp { SZ.v rows <= max_blocks * max_threads })
     (l    : szp)
     (l_out : sz { SZ.v l_out == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) })
     (#lin  : layout2 (SZ.v rows) (SZ.v l))     {| ctlayout lin  |}
     (#lout : layout2 (SZ.v rows) (SZ.v l_out)) {| ctlayout lout |}
     (input  : array2 et lin  { is_global input  })
     (output : array2 et lout { is_global output })
     (#sx   : EM.chest2 et (SZ.v rows) (SZ.v l))
     (#sout : EM.chest2 et (SZ.v rows) (SZ.v l_out))
     (#fIn  : perm)
     preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
     requires
       on gpu_loc (output |-> sout) **
       pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
       pure (SZ.v rows * SZ.v l_out <= max_blocks * max_threads)
     ensures
       on gpu_loc (output |->
         windowreduce_result m sx (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) (SZ.v l_out))

inline_for_extraction noextract
val windowreduce : windowreduce_ty

(* ── Monomorphic wrapper: max-pool (#41) ─────────────────────────────── *)

(* TODO: declare the [windowreduce_max_f32] entry whose ensures
 * clause is exactly [maxpool1d_post sx sout'] from
 * [Kuiper.Spec.Pool1D].  The bridge proof is a pure F* fact about
 * [cmonoid_fmax_f32] equalling the [fmax]-fold over in-bounds taps. *)

(* ── Monomorphic wrapper: avg-pool (#44) ─────────────────────────────── *)

(* TODO: declare the [windowreduce_plus_f32] entry whose ensures
 * clause is the [%~]-form [avgpool1d_post] modulo the
 * caller-supplied [inv_k] scaling.  The bridge proof composes the
 * abstract [cmonoid_fadd_f32] reduction with the
 * [Kuiper.Approximates.a_add] %~-lemma chain. *)
