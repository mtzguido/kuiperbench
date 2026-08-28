module Kuiper.Kernel.TriangularMatmul

(* Fused triangular matrix multiplication.

   The shortened reduction is valid under the explicit triangular-input
   precondition.  The result remains specified as the corresponding triangle
   of the ordinary dense matmul. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module MS = Kuiper.Spec.GEMM

let is_upper_triangular
  (#t : Type0) {| scalar t |}
  (#n : nat)
  (a : chest2 t n n)
  : prop
  = forall (i j : natlt n).
      j < i ==> acc2 a i j == zero

let is_lower_triangular
  (#t : Type0) {| scalar t |}
  (#n : nat)
  (a : chest2 t n n)
  : prop
  = forall (i j : natlt n).
      i < j ==> acc2 a i j == zero

val upper_matmul_is_upper
  (#n : nat)
  (a b : chest2 real n n)
  : Lemma
      (requires is_upper_triangular a /\ is_upper_triangular b)
      (ensures is_upper_triangular (MS.matmul a b))

val lower_matmul_is_lower
  (#n : nat)
  (a b : chest2 real n n)
  : Lemma
      (requires is_lower_triangular a /\ is_lower_triangular b)
      (ensures is_lower_triangular (MS.matmul a b))

inline_for_extraction noextract
fn upper_triangular_matmul
  (#et:Type0) {| floating et, real_like et |}
  (n : szp { n * n <= max_blocks * max_threads })
  (#lA : layout2 n n) {| ctlayout lA |}
  (gA : array2 et lA { is_global gA })
  (#lB : layout2 n n) {| ctlayout lB |}
  (gB : array2 et lB { is_global gB })
  (#lC : layout2 n n) {| ctlayout lC |}
  (gC : array2 et lC { is_global gC })
  (#sA #sB #sC : chest2 et n n)
  (#rA #rB : chest2 real n n)
  preserves
    cpu **
    on gpu_loc (gA |-> sA) **
    on gpu_loc (gB |-> sB)
  requires
    on gpu_loc (gC |-> sC) **
    (* Note: we do not require the actual float contents to be the zero float.
    Just approximating zero is enough. Those elements will be skipped anyway. *)
    pure (
      sA %~ rA /\
      sB %~ rB /\
      is_upper_triangular rA /\
      is_upper_triangular rB)
  ensures
    exists* sgC'.
       on gpu_loc (gC |-> sgC') **
       pure (sgC' %~ MS.matmul rA rB)

(* The lower-triangular operation is the upper-triangular operation on
   transposed views with the operands reversed.  The views change only the
   logical layout; they do not copy or rearrange the matrices. *)
inline_for_extraction noextract
fn lower_triangular_matmul
  (#et:Type0) {| floating et, real_like et |}
  (n : szp { n * n <= max_blocks * max_threads })
  (gA : array2 et (l2_row_major n n) { is_global gA })
  (gB : array2 et (l2_row_major n n) { is_global gB })
  (gC : array2 et (l2_row_major n n) { is_global gC })
  (#sA #sB #sC : chest2 et n n)
  (#rA #rB : chest2 real n n)
  preserves
    cpu **
    on gpu_loc (gA |-> sA) **
    on gpu_loc (gB |-> sB)
  requires
    on gpu_loc (gC |-> sC) **
    pure (
      sA %~ rA /\
      sB %~ rB /\
      is_lower_triangular rA /\
      is_lower_triangular rB)
  ensures
    exists* sgC'.
       on gpu_loc (gC |-> sgC') **
       pure (sgC' %~ MS.matmul rA rB)
