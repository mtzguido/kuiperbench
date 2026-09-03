module Kuiper.KB.TrilMatmul

(* KernelBench L1 #15: Matmul for lower-triangular matrices.

   PyTorch reference:
     C = torch.tril(torch.matmul(A, B))      # A, B square (N, N)
   Output the lower-triangular part of the N x N matmul (strictly-upper
   entries zeroed).

   KernelBench supplies lower-triangular operands.  The verified kernel uses
   that precondition to compute only the retained reduction range. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module TM = Kuiper.Kernel.TriangularMatmul

fn tril_matmul_f32
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
    pure (
      sA %~ rA /\
      sB %~ rB /\
      TM.is_lower_triangular rA /\
      TM.is_lower_triangular rB)
  ensures
    (exists* (sy' : chest2 f32 n n).
       on gpu_loc (y |-> sy') **
       pure (sy' %~ MS.matmul rA rB))

fn tril_matmul_alloc_f32
  (n : szp {
     SZ.v n * SZ.v n <= max_blocks * max_threads /\
     SZ.fits (SZ.v n * SZ.v n) })
  (gA : array2 f32 (l2_row_major n n) { is_global gA })
  (gB : array2 f32 (l2_row_major n n) { is_global gB })
  (#sA #sB : chest2 f32 n n)
  (#rA #rB : chest2 real n n)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> sA) ** on gpu_loc (gB |-> sB)
  requires
    pure (sA %~ rA /\ sB %~ rB /\
          TM.is_lower_triangular rA /\ TM.is_lower_triangular rB)
  returns y : array2 f32 (l2_row_major n n)
  ensures
    exists* (sy : chest2 f32 n n).
      on gpu_loc (y |-> sy) ** pure (sy %~ MS.matmul rA rB)
