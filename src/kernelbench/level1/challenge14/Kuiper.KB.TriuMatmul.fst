module Kuiper.KB.TriuMatmul

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module TM = Kuiper.Kernel.TriangularMatmul

inline_for_extraction noextract
fn triu_matmul_f32_impl
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
    on gpu_loc (y |-> sy) **
    pure (
      sA %~ rA /\
      sB %~ rB /\
      TM.is_upper_triangular rA /\
      TM.is_upper_triangular rB)
  ensures
    exists* (sy' : chest2 f32 n n).
      on gpu_loc (y |-> sy') **
      pure (sy' %~ MS.matmul rA rB)
{
  TM.upper_triangular_matmul n gA gB y
    #sA #sB #sy #rA #rB;
  ()
}

let triu_matmul_f32 = triu_matmul_f32_impl
