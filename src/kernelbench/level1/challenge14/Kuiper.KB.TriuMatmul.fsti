module Kuiper.KB.TriuMatmul

(* KernelBench L1 #14: upper-triangular part of a square matmul.

   The exported entry composes two verified GPU kernels:
     1. y := A @ B
     2. y := triu y, in place

   The postcondition covers the whole KernelBench operation; the C++ bridge
   performs no mathematical post-processing. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

let triu_matmul_post
  (#t:Type0) {| scalar t, real_like t |}
  (n : nat)
  (rA rB : chest2 real n n)
  (sy' : chest2 t n n)
  : prop
  = forall (i j : natlt n).
      acc2 sy' i j %~
        (if i <= j then acc2 (MS.matmul rA rB) i j else 0.0R)

fn triu_matmul_f32
  (n : szp {
     SZ.v n * SZ.v n <= max_blocks * max_threads /\
     SZ.fits (SZ.v n * SZ.v n) })
  (gA : array2 f32 (l2_row_major n n) { is_global gA })
  (gB : array2 f32 (l2_row_major n n) { is_global gB })
  (y  : array2 f32 (l2_row_major n n) { is_global y  })
  (#sA #sB #sy : chest2 f32 n n)
  (#rA #rB : chest2 real n n)
  preserves
    cpu **
    on gpu_loc (gA |-> sA) **
    on gpu_loc (gB |-> sB)
  requires
    on gpu_loc (y  |-> sy) **
    pure (reveal sA %~ reveal rA /\ reveal sB %~ reveal rB)
  ensures
    (exists* (sy' : chest2 f32 n n).
       on gpu_loc (y |-> sy') **
       pure (triu_matmul_post n (reveal rA) (reveal rB) sy'))
