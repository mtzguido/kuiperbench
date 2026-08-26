module Kuiper.Kernel.HReduce.Min

#lang-pulse

open Kuiper
open Kuiper.Math.Fmin
open Kuiper.Tensor
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module Seq = FStar.Seq

(* ── Spec functions for reduce_batched_min ─────────────────────────────────
   Per-row partial fmin fold over an [(rows, cols)] row-major
   chest2.  IEEE-754 [fmin] on [f32] is bit-exactly associative+
   commutative on this carrier (axiomatized in [Kuiper.Math.Fmin]),
   so the spec is exact equality (no [%~]).

   Mirrors [Kuiper.Kernel.HReduce.row_reduce_partial] with [add]
   replaced by [fmin] and seed [zero] by [pos_inf]. *)

[@@"opaque_to_smt"]
let rec row_fmin_partial
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot f32
  (decreases k)
  = if k = 0 then pos_inf
    else fmin (row_fmin_partial sx r (k - 1)) (acc2 sx r (k - 1))

val row_fmin_partial_zero
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_fmin_partial sx r 0 == pos_inf)
          [SMTPat (row_fmin_partial sx r 0)]

val row_fmin_partial_succ
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma (row_fmin_partial sx r (k + 1) ==
           fmin (row_fmin_partial sx r k) (acc2 sx r k))
          [SMTPat (row_fmin_partial sx r (k + 1))]

(* Full reduction = [seq_fmin] of the row. *)
val row_fmin_eq_seq_fmin
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_fmin_partial sx r cols == seq_fmin (EM.ematrix_row sx r))

let row_fmin
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : GTot f32
  = row_fmin_partial sx r cols

let seq_reduce_rows_fmin
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  : GTot (Seq.lseq f32 rows)
  = Seq.init_ghost rows (fun r -> row_fmin sx r)

(* ── reduce_batched_min: one thread per row, simple serial reduction ──── *)

inline_for_extraction noextract
val reduce_batched_min_f32
     (rows : szp { SZ.v rows <= max_blocks * max_threads })
     (cols : szp)
     (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
     (#lout : layout1 (SZ.v rows))             {| ctlayout lout |}
     (x      : array2 f32 lin  { is_global x      })
     (output : array1 f32 lout { is_global output })
     (#sx   : erased (EM.chest2 f32 (SZ.v rows) (SZ.v cols)))
     (#sout : chest1 f32 (SZ.v rows))
  : stt unit
      (cpu **
       on gpu_loc (x |-> sx) **
       on gpu_loc (output |-> sout))
      (fun _ ->
       cpu **
       on gpu_loc (x |-> sx) **
       on gpu_loc (output |-> seq_to_chest1 (seq_reduce_rows_fmin sx)))
