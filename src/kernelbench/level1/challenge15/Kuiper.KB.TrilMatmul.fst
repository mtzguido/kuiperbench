module Kuiper.KB.TrilMatmul

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module TM = Kuiper.Kernel.TriangularMatmul

inline_for_extraction noextract
fn tril_matmul_f32_impl
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
      TM.is_lower_triangular rA /\
      TM.is_lower_triangular rB)
  ensures
    exists* (sy' : chest2 f32 n n).
      on gpu_loc (y |-> sy') **
      pure (sy' %~ MS.matmul rA rB)
{
  TM.lower_triangular_matmul n gA gB y
    #sA #sB #sy #rA #rB;
  ()
}

let tril_matmul_f32 = tril_matmul_f32_impl

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
{
  let y = alloc0 #f32 (n *^ n) (l2_row_major n n);
  tril_matmul_f32 n gA gB y #sA #sB #_ #rA #rB;
  y
}
