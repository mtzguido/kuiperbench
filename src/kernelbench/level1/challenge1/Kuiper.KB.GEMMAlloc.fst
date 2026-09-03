module Kuiper.KB.GEMMAlloc

(* Self-allocating row-major GEMM entries shared by KernelBench L1
   #1/#2/#4/#6/#7/#8/#9/#13.  Allocation and the GEMM launch are one verified
   composition; the C++ boundary only validates the erased refinements and
   wraps the returned CUDA allocation. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module MS = Kuiper.Spec.GEMM
module K1 = Kuiper.Kernel.GEMM.Naive1
module K3 = Kuiper.Kernel.GEMM.Naive3
module Klas1 = Klas.GEMM.Naive1
module Klas3 = Klas.GEMM.Naive3
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn guard_gemm_sizes
  (m n k : szp)
  norewrite
  requires emp
  ensures pure (SZ.fits (m * k) /\ SZ.fits (k * n) /\ SZ.fits (m * n))
{
  let maxu32 : sz = SZ.uint_to_t 4294967295;
  dguard (k <=^ (maxu32 /^ m));
  dguard (n <=^ (maxu32 /^ k));
  dguard (n <=^ (maxu32 /^ m));
}

inline_for_extraction noextract
fn guard_naive3_size
  (m n k : szp)
  norewrite
  requires emp
  ensures pure (SZ.fits (m * k) /\ SZ.fits (k * n) /\ SZ.fits (m * n) /\
                K3.size_req m n k)
{
  guard_gemm_sizes m n k;
  let mn : szp = m *^ n;
  let bound : szp = max_blocks *^ max_threads;
  dguard (mn <=^ bound);
}

inline_for_extraction noextract
fn guard_naive1_size
  (m n k : szp)
  norewrite
  requires emp
  ensures pure (SZ.fits (m * k) /\ SZ.fits (k * n) /\ SZ.fits (m * n) /\
                m * n <= max_blocks)
{
  guard_gemm_sizes m n k;
  let mn : szp = m *^ n;
  dguard (mn <=^ max_blocks);
}

fn gemm_naive3_alloc_f32
  (m n k : szp)
  (a : array2 f32 (l2_row_major m k) { is_global a })
  (b : array2 f32 (l2_row_major k n) { is_global b })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#sA : chest2 f32 m k)
  (#sB : chest2 f32 k n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sA ** b |-> Frac fB sB)
  requires pure (sA %~ rA /\ sB %~ rB)
  returns c : array2 f32 (l2_row_major m n)
  ensures
    exists* (sC : chest2 f32 m n).
      on gpu_loc (c |-> sC) ** pure (sC %~ MS.matmul rA rB)
{
  guard_naive3_size m n k;
  let c = alloc0 #f32 (m *^ n) (l2_row_major m n);
  Klas3.spec f32 l2_row_major l2_row_major l2_row_major
    m n k a b c rA rB;
  c
}

fn gemm_naive1_alloc_f32
  (m n k : szp)
  (a : array2 f32 (l2_row_major m k) { is_global a })
  (b : array2 f32 (l2_row_major k n) { is_global b })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#sA : chest2 f32 m k)
  (#sB : chest2 f32 k n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sA ** b |-> Frac fB sB)
  requires pure (sA %~ rA /\ sB %~ rB)
  returns c : array2 f32 (l2_row_major m n)
  ensures
    exists* (sC : chest2 f32 m n).
      on gpu_loc (c |-> sC) ** pure (sC %~ MS.matmul rA rB)
{
  guard_naive1_size m n k;
  let c = alloc0 #f32 (m *^ n) (l2_row_major m n);
  Klas1.spec_2d f32 l2_row_major l2_row_major l2_row_major
    m n k a b c rA rB;
  c
}
