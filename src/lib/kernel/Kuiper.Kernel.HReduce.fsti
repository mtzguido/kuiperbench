module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common { seq_map }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix

// TODO: generalize operation? It currently always uses `add`
// from the scalar class.

(* Could we use this instead of approx2? *)
instance approx_function_can_approximate
  (dom1 dom2 cod1 cod2 : Type)
  {| can_approximate dom1 dom2, can_approximate cod1 cod2 |}
  : can_approximate (dom1 -> cod1) (dom2 -> cod2) = {
  approximates = (fun f g -> forall x y. x %~ y ==> f x %~ g y);
}

inline_for_extraction noextract
type reduce_ty (et : Type0) {| scalar et, real_like et |} =
  fn (pre_map : et -> et)
     (pre_map_r : real -> real { pre_map %~ pre_map_r })
     (nth : szp { nth <= max_threads })
     (lena : sz)
     (#l : layout1 lena) {| ctlayout l |}
     (a : array1 et l { is_global a })
     (#va : chest1 et lena)
     (vr : chest1 real lena)
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr) **
    pure (SZ.fits (lena + nth)) // Almost impossible to falsify
  returns
    res : et
  ensures
    pure (res %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))

inline_for_extraction noextract
val reduce (#et:Type0) {| scalar et, real_like et |} : reduce_ty et

(* ── Spec functions for reduce_batched ───────────────────────────────── *)

(* Partial reduction of row r, processing elements 0..k-1. Opaque to SMT;
   use row_reduce_partial_zero / row_reduce_partial_succ lemmas instead. *)
[@@"opaque_to_smt"]
let rec row_reduce_partial
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (#rows #cols : nat)
  (sx : EM.chest2 et rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot et
  (decreases k)
  = if k = 0 then zero
    else row_reduce_partial pre_map sx r (k - 1) `add` pre_map (acc2 sx r (k - 1))

(* Unfolding lemmas for row_reduce_partial (needed because it's opaque). *)
val row_reduce_partial_zero
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (#rows #cols : nat)
  (sx : EM.chest2 et rows cols)
  (r : natlt rows)
  : Lemma (row_reduce_partial pre_map sx r 0 == zero)
          [SMTPat (row_reduce_partial pre_map sx r 0)]

val row_reduce_partial_succ
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (#rows #cols : nat)
  (sx : EM.chest2 et rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma (row_reduce_partial pre_map sx r (k + 1) ==
           row_reduce_partial pre_map sx r k `add` pre_map (acc2 sx r k))
          [SMTPat (row_reduce_partial pre_map sx r (k + 1))]

(* Full reduction of row r. *)
let row_reduce
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (#rows #cols : nat)
  (sx : EM.chest2 et rows cols)
  (r : natlt rows)
  : GTot et
  = row_reduce_partial pre_map sx r cols

(* Sequence of per-row reductions. *)
let seq_reduce_rows
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (#rows #cols : nat)
  (sx : EM.chest2 et rows cols)
  : GTot (lseq et rows)
  = Seq.init_ghost rows (fun r -> row_reduce pre_map sx r)

(* ── reduce_batched: one thread per row ──────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)              {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (#sx   : EM.chest2 et rows cols)
  (#sout : chest1 et rows)
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout)
  ensures
    on gpu_loc (output |-> seq_to_chest1 (seq_reduce_rows pre_map sx))
