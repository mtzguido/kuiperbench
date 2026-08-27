module Kuiper.KB.TrilMatmul

(* KernelBench L1 #15: Matmul for lower-triangular matrices.

   PyTorch reference:
     C = torch.tril(torch.matmul(A, B))      # A, B square (N, N)
   Output the lower-triangular part of the N x N matmul (strictly-upper
   entries zeroed).

   The whole computation is performed by VERIFIED Kuiper kernels; there
   is NO unverified [torch::tril] post-processing.  Pipeline (GPU-only,
   two launches, in-place on the output buffer):
     1. GEMM (Kahan / Naive3) : y := A @ B                 (N x N)
        (Kuiper.Kernel.GEMM.Naive3.mmcomb_gpu_approx)
     2. lower-triangular mask : y := s_tril y              (in place)
        (Kuiper.Kernel.Tril.tril)

   Because the GEMM uses Kahan summation it only admits an APPROXIMATE
   real-number spec, so the exported postcondition relates each output
   matrix entry to the real-valued lower-triangular matmul via [%~].

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Per-element APPROXIMATE functional postcondition: each output-matrix
   entry [(i, j)] approximates the lower-triangular real matmul entry: the
   real matmul value [acc2 (matmul rA rB) i j] on/below the diagonal, and
   exactly [0] strictly above it. *)
let tril_matmul_post
  (#t:Type0) {| scalar t, real_like t |}
  (n : nat)
  (rA rB : EM.chest2 real n n)
  (sy' : EM.chest2 t n n)
  : prop
  = forall (i j : natlt n).
      acc2 sy' i j %~
        (if j <= i then acc2 (MS.matmul rA rB) i j else 0.0R)

inline_for_extraction noextract
type tril_matmul_ty (t:Type0) {| floating t, real_like t, floating_real_like t |} =
  fn (n : szp {
        SZ.v n * SZ.v n <= max_blocks * max_threads /\
        SZ.fits (SZ.v n * SZ.v n) })
     (gA : array2 t (l2_row_major n n) { is_global gA })
     (gB : array2 t (l2_row_major n n) { is_global gB })
     (y  : array2 t (l2_row_major n n) { is_global y  })
     (#sA #sB #sy : EM.chest2 t n n)
     (#rA #rB : EM.chest2 real n n)
     requires
       cpu **
       on gpu_loc (gA |-> sA) **
       on gpu_loc (gB |-> sB) **
       on gpu_loc (y  |-> sy) **
       pure (reveal sA %~ reveal rA /\ reveal sB %~ reveal rB)
     ensures
       cpu **
       on gpu_loc (gA |-> sA) **
       on gpu_loc (gB |-> sB) **
       (exists* (sy' : EM.chest2 t n n).
          on gpu_loc (y |-> sy') **
          pure (tril_matmul_post n (reveal rA) (reveal rB) sy'))

val tril_matmul_f32 : tril_matmul_ty f32
