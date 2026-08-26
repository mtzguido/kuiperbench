module Kuiper.Kernel.RowDivExp

#lang-pulse

open Kuiper
module RB = Kuiper.Kernel.RowBroadcast
open Kuiper.EMatrix
open Kuiper.Tensor

inline_for_extraction noextract
let div_exp
  (#t : Type0) {| floating t, real_like t, floating_real_like t |}
  (x s : t) : t = div (fexp x) s

let s_row_div_exp
  (#t : Type0) {| floating t, real_like t, floating_real_like t |}
  (#m #n : nat)
  (a : chest1 t m) (b : chest2 t m n)
  : chest2 t m n
  = RB.s_row_broadcast (div_exp #t) a b

type row_div_exp_ty
  (t : Type0) {| floating t, real_like t, floating_real_like t |} =
  fn
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la : layout1 m) {| ctlayout la |}
  (a : array1 t la)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#fA : perm)
  (#sa : chest1 t m)
  (#sb : chest2 t m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sa)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_row_div_exp sa sb)

inline_for_extraction noextract
val row_div_exp
  (t : Type0) {| floating t, real_like t, floating_real_like t |}
  : row_div_exp_ty t
