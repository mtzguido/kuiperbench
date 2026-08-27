module Kuiper.Kernel.RowBroadcast2

(* Generic 2-input row-broadcast 2D map.
   Computes [B[i, j] := f (B[i, j]) (A1[i]) (A2[i])] in-place, where
   [A1] and [A2] are [Array1]s of length m and [B] is an [Array2]
   of shape m × n. *)

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor

let s_row_broadcast2
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (#m #n : nat)
  (a1 a2 : chest1 t m) (b : chest2 t m n)
  : chest2 t m n
  = mk2 fun i j -> f (acc2 b i j) (acc1 a1 i) (acc1 a2 i)

inline_for_extraction noextract
fn row_broadcast2
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la1 : layout1 m) {| ctlayout la1 |}
  (a1 : array1 t la1)
  (#la2 : layout1 m) {| ctlayout la2 |}
  (a2 : array1 t la2)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#_ : squash (is_global a1))
  (#_ : squash (is_global a2))
  (#_ : squash (is_global b))
  (#fA1 #fA2 : perm)
  (#sa1 #sa2 : chest1 t m)
  (#sb : chest2 t m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a1 |-> Frac fA1 sa1) ** on gpu_loc (a2 |-> Frac fA2 sa2)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_row_broadcast2 f sa1 sa2 sb)
