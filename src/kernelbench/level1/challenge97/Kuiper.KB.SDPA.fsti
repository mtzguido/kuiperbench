module Kuiper.KB.SDPA

(* KernelBench L1 #97 — Scaled Dot-Product Attention.

     out = softmax((Q @ K^T) / sqrt(D)) @ V        (no mask, no dropout)

   The public entry and functional specification retain the original
   (B,H,S,D) row-major tensors.  Inside the verified entry, Kuiper proves the
   zero-copy collapse (B,H,S,D) -> (B*H,S,D), then proves that the same K bytes
   can be viewed as a page-wise transposed column-major tensor for the first
   GEMM.  Neither flattening is delegated to the host bridge.

   This module is a verified ORCHESTRATOR.  It does NOT fuse anything; it
   chains four already-verified Kuiper primitives over the SAME GPU buffers:

     1. batched_gemm_f32 : gScores := Q @ K^T            (exact float)
     2. smul_fw_f32      : gScores *= 1/sqrt(D)          (exact float)
     3. row_softmax      : gScores := softmax_S(gScores) (the %~ step)
     4. batched_gemm_f32 : gOut    := gScores @ V        (exact float)

   Steps 2 and 3 reinterpret the (BH,S,S) Array3 buffer as a flat Array1
   (for the scalar multiply) resp. a (BH*S, S) Array2 (for the row softmax)
   via pure ghost re-interpretations (no data movement); see the reshape
   lemmas in the .fst, which mirror Kuiper.KB.MatmulND.

   Functional spec (stated directly over real tensors approximated by the
   three f32 inputs):

     out %~ softmax((Q @ K^T) * (1 / sqrt(D))) @ V

   The two GEMMs, scaling, and softmax are all connected to their real
   counterparts by the corresponding [%~] laws.  No floating intermediate
   appears in the public functional claim.

   The direct real scale proof uses the temporary
   [Kuiper.KB.Compat.RsqrtApprox.rsqrt_approx] compatibility assumption. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l3_batched_row_major, l4_batched_row_major }
module SZ = Kuiper.SizeT
module SM = Kuiper.Spec.Softmax
module RealSqrt = FStar.Math.Sqrt
open Kuiper.KB.BatchedGEMM { batched_matmul }

(* Private extraction helper: the public entry computes this value itself. *)
inline_for_extraction noextract
val sdpa_scale_f32 (d : szp) : f32

(* Logical transpose of the last two axes on every batch page. *)
[@@erasable]
val transpose_pages
  (#et : Type) (#bh #rows #cols : nat)
  (m : chest3 et bh rows cols)
  : chest3 et bh cols rows

(* Elementwise scalar multiply of a 3-D tensor. *)
let mscale
  (#t:Type0) {| scalar t |}
  (#d0 #d1 #d2 : nat)
  (c : t)
  (m : chest3 t d0 d1 d2)
  : chest3 t d0 d1 d2
  = mk3 (fun i j k -> mul c (acc3 m i j k))

(* Page-wise per-row real softmax of a (bh,s,s) real tensor: page [p] is the
   row-softmax of page [p].  Defined in exactly the same shape as
   [Kuiper.Kernel.RowSoftmax.row_softmax_real] applied to each page, so the
   orchestrator's correspondence proof reduces both sides to the same
   [acc1 (softmax_real (chest2_row ..)) ..] cell. *)
let softmax_pages
  (#bh #s : nat)
  (rm : chest3 real bh s s)
  : chest3 real bh s s
  = mk3 (fun p i j ->
      acc1 (SM.softmax_real (chest2_row (slice_page rm p) i)) j)

let real_sdpa_scale (d : pos) : real =
  RealSqrt.rsqrt (FStar.Real.of_int d)

let real_sdpa
  (#bh #s #d : pos)
  (rQ : chest3 real bh s d)
  (rK : chest3 real bh s d)
  (rV : chest3 real bh s d)
  : chest3 real bh s d =
  batched_matmul
    (softmax_pages
      (mscale
        (real_sdpa_scale d)
        (batched_matmul rQ (transpose_pages rK))))
    rV

(* Row-major collapse of the original batch and head axes.  The implementation
   uses this same serialization to reinterpret each Array4 as Array3 without
   moving data. *)
let flatten_bh
  (#et : Type) (#b #h #s #d : nat)
  (x : chest4 et b h s d)
  : chest3 et (b * h) s d =
  from_seq (l3_batched_row_major (b * h) s d)
    (to_seq (l4_batched_row_major b h s d) x)

let unflatten_bh
  (#et : Type) (#b #h #s #d : nat)
  (x : chest3 et (b * h) s d)
  : chest4 et b h s d =
  from_seq (l4_batched_row_major b h s d)
    (to_seq (l3_batched_row_major (b * h) s d) x)

(* Rank-4 top-level semantics.  This states explicitly that every (b,h) page
   of the original tensors is the corresponding page of the verified batched
   attention computation. *)
let real_sdpa4
  (#b #h #s #d : pos)
  (rQ : chest4 real b h s d)
  (rK : chest4 real b h s d)
  (rV : chest4 real b h s d)
  : chest4 real b h s d =
  unflatten_bh (real_sdpa (flatten_bh rQ) (flatten_bh rK) (flatten_bh rV))

fn sdpa_f32
  (b h s d : szp)
  (gQ : array4 f32 (l4_batched_row_major b h s d) { is_global gQ })
  (gK : array4 f32 (l4_batched_row_major b h s d) { is_global gK })
  (gV : array4 f32 (l4_batched_row_major b h s d) { is_global gV })
  (#sQ #sK #sV : chest4 f32 b h s d)
  (rQ rK rV : erased (chest4 real b h s d))
  (#fQ #fK #fV : perm)
  preserves
    cpu **
    on gpu_loc
      (gQ |-> Frac fQ sQ **
       gK |-> Frac fK sK **
       gV |-> Frac fV sV) **
    pure (sQ %~ rQ /\ sK %~ rK /\ sV %~ rV)
  requires
    pure (
      SZ.fits (b * h) /\
      s * s <= max_blocks * max_threads /\
      SZ.fits ((b * h) * (s * d)) /\
      SZ.fits ((b * h) * (d * s)) /\
      SZ.fits ((b * h) * (s * s)) /\
      (b * h) * s * s <= max_blocks * max_threads /\
      (b * h) * s <= max_blocks /\
      ((b * h) * s) * s <= max_blocks * max_threads /\
      (b * h) * (s * d) <= max_blocks * max_threads
    )
  returns gOut : array4 f32 (l4_batched_row_major b h s d)
  ensures
    (exists* (eOut : chest4 f32 b h s d).
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ real_sdpa4 rQ rK rV))
