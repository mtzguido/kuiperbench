module Kuiper.KB.Matmul4D

(* KernelBench L1 #11 — 4-D tensor (b,i,j,l) @ (l,k) matrix -> (b,i,j,k),
   i.e. einsum("bijl,lk->bijk").  The SAME matrix B multiplies every
   (b,i,j) slice:  C[b,i,j,k] = sum_l A[b,i,j,l] * B[l,k].

   Instead of an UNVERIFIED host-side flatten (numel/reshape in C++), we
   collapse the leading two dims of the row-major Array4 (b,i,j,l) into a
   row-major Array3 (b*i,j,l) INSIDE the verified kernel — a pure ghost
   re-interpretation of the SAME GPU buffer (no data movement) — and then
   CALL the already-verified challenge-10 kernel [Kuiper.KB.MatmulND] on the
   (b*i,j,l) @ (l,k) product.  MatmulND internally collapses (b*i,j) -> b*i*j
   and runs the layout-polymorphic Kahan-summed Naive3 GEMM, so no GEMM /
   szp-gap / summation reasoning is re-derived here.

   Functional spec: the 4-D output equals the DIRECT 4-D matmul product
   [ematmul4] of the input with B, in the real-number approximation [%~] of the
   kernel.  [ematmul4] multiplies every (b,i)-slice A[b,i,:,:] by B pointwise,
   so the interface never mentions any flatten.

   Zero assume · zero magic · zero admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Shape
module EMatrix = Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM

(* Local row-major 4-D layout: the direct analogue of [l3_batched_row_major].
   Defined locally (rather than in the core library) since [matmul4d] only
   needs the ghost [Array4] ops [lower]/[raise']/[from_array]/[core], which do
   not require a [ctlayout] instance.  Full automatically ([ulen == sizeof]). *)
let l4_row_major (d0 d1 d2 d3 : nat) : tlayout (d0 @| d1 @| d2 @| d3 @| INil) =
  pack <|
  major_on 0 d0 <|
  major_on 0 d1 <|
  major_on 0 d2 <|
  major_on 0 d3 <|
  lunit

(* The (j,l) matrix obtained by fixing the leading two indices (b_,i_) of a
   4-D tensor.  This is the "row pair slice" A[b_,i_,:,:] that the per-(b,i,j)
   batched matmul multiplies by B.  (acc4 needs four indices, so the
   maintainer's informal [acc2 a b_ i_] is spelled out here as a 2-D slice.) *)
inline_for_extraction noextract
let row_pair_slice (#et:Type) (#b #i #j #l:nat)
  (a : chest4 et b i j l) (b_:natlt b) (i_:natlt i)
  : chest2 et j l
  = mk2 (fun (j_:natlt j) (l_:natlt l) -> acc4 a b_ i_ j_ l_)

(* DIRECT 4-D matmul product (einsum "bijl,lk->bijk"): the SAME matrix [bm]
   multiplies every (b_,i_) slice.  Defined pointwise so the functional spec
   does not have to mention any flatten. *)
inline_for_extraction noextract
let ematmul4 (#et:Type) {| Kuiper.scalar et |} (#b #i #j #l #k:nat)
  (a : chest4 et b i j l) (bm : chest2 et l k)
  : chest4 et b i j k
  = mk4 (fun (b_:natlt b) (i_:natlt i) (j_:natlt j) (k_:natlt k) ->
      acc2 (MS.matmul (row_pair_slice a b_ i_) bm) j_ k_)

fn matmul4d_f32
  (b i j l k : szp)
  (gA : array4 f32 (l4_row_major b i j l) { is_global gA })
  (gB : array2 f32 (l2_row_major l k)     { is_global gB })
  (gC : array4 f32 (l4_row_major b i j k) { is_global gC })
  (rA : chest4 real b i j l)
  (rB : chest2 real l k)
  (#eA : chest4 f32 b i j l)
  (#eB : chest2 f32 l k)
  (#eC : chest4 f32 b i j k)
  (#fA #fB : perm)
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (eA %~ rA) **
    pure (eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest4 f32 b i j k).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ ematmul4 rA rB))
