module Kuiper.Kernel.HReduce.Max.RowFmax

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Math.Fmax
module EM = Kuiper.EMatrix
module Seq = FStar.Seq

(* ── Ghost spec helpers for per-row [fmax] reduction ───────────────────────
   These were previously part of [Kuiper.Kernel.HReduce.Max] but were
   removed upstream when the batched max kernel was refactored.  They are
   pure ghost folds with no runtime content and are re-introduced here so
   downstream modules (argmax-value bridge, etc.) keep working.

   Per-row partial fmax fold over an [(rows, cols)] row-major chest2.
   IEEE-754 [fmax] on [f32] is bit-exactly associative+commutative on this
   carrier (axiomatized in [Kuiper.Math.Fmax]), so the spec is exact
   equality (no [%~]). *)

[@@"opaque_to_smt"]
let rec row_fmax_partial
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot f32
  (decreases k)
  = if k = 0 then neg_inf
    else fmax (row_fmax_partial sx r (k - 1)) (acc2 sx r (k - 1))

val row_fmax_partial_zero
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_fmax_partial sx r 0 == neg_inf)
          [SMTPat (row_fmax_partial sx r 0)]

val row_fmax_partial_succ
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma (row_fmax_partial sx r (k + 1) ==
           fmax (row_fmax_partial sx r k) (acc2 sx r k))
          [SMTPat (row_fmax_partial sx r (k + 1))]

(* Full reduction = [seq_fmax] of the row. *)
val row_fmax_eq_seq_fmax
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_fmax_partial sx r cols == seq_fmax (EM.ematrix_row sx r))

let row_fmax
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  (r : natlt rows)
  : GTot f32
  = row_fmax_partial sx r cols

let seq_reduce_rows_fmax
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  : GTot (Seq.lseq f32 rows)
  = Seq.init_ghost rows (fun r -> row_fmax sx r)
