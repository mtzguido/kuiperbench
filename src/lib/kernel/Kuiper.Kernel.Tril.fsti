module Kuiper.Kernel.Tril

(* Generic in-place lower-triangular mask kernel.

   For an [m x n] matrix [b] (held as an [array2], any layout) masks the
   strictly-upper-triangular entries to zero, in place:

       b[i, j] := if j <= i then b[i, j] else 0.

   One thread per element ([m * n] threads).  The result is the [chest2]
   [s_tril sb], defined directly by [mk2] — no flattening, no flat indices.

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor
(* Functional spec: lower-triangular mask of [b], as an [chest2]. *)
let s_tril
  (#t:Type0) {| scalar t |}
  (#m #n : nat)
  (b : chest2 t m n)
  : chest2 t m n
  = mk2 fun i j -> if j <= i then acc2 b i j else zero

inline_for_extraction noextract
fn tril
  (t:Type0) {| scalar t |}
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#_ : squash (is_global b))
  (#sb : erased (chest2 t m n))
  preserves cpu
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_tril sb)
