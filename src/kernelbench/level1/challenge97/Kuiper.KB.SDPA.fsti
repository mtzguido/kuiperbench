module Kuiper.KB.SDPA

(* KernelBench L1 #97 — Scaled Dot-Product Attention.

     out = softmax((Q @ K^T) / sqrt(D)) @ V        (no mask, no dropout)

   Q, K, V have shape (B, H, S, D); the batch+head dims are flattened to
   BH = B*H, giving (BH, S, D) tensors.  K is supplied ALREADY TRANSPOSED
   per page as (BH, D, S) (the host bridge materializes K^T, an unverified
   PyTorch transpose — same pattern as L1 #17).

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

   The direct real scale proof uses the temporary [rsqrt_approx]
   compatibility assumption documented in the repository patch. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l3_batched_row_major }
module SZ = Kuiper.SizeT
module SM = Kuiper.Spec.Softmax
module RealSqrt = FStar.Math.Sqrt
open Kuiper.KB.BatchedGEMM { batched_matmul }

(* Private extraction helper: the public entry computes this value itself. *)
inline_for_extraction noextract
val sdpa_scale_f32 (d : szp) : f32

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
  (rKT : chest3 real bh d s)
  (rV : chest3 real bh s d)
  : chest3 real bh s d =
  batched_matmul
    (softmax_pages
      (mscale (real_sdpa_scale d) (batched_matmul rQ rKT)))
    rV

fn sdpa_f32
  (bh s d : szp)
  (gQ  : array3 f32 (l3_batched_row_major bh s d) { is_global gQ })
  (gKT : array3 f32 (l3_batched_row_major bh d s) { is_global gKT })
  (gV  : array3 f32 (l3_batched_row_major bh s d) { is_global gV })
  (gScores : array3 f32 (l3_batched_row_major bh s s) { is_global gScores })
  (gOut : array3 f32 (l3_batched_row_major bh s d) { is_global gOut })
  (#sQ  : chest3 f32 bh s d)
  (#sKT : chest3 f32 bh d s)
  (#sV  : chest3 f32 bh s d)
  (#sScores0 : chest3 f32 bh s s)
  (#sOut0 : chest3 f32 bh s d)
  (rQ : erased (chest3 real bh s d))
  (rKT : erased (chest3 real bh d s))
  (rV : erased (chest3 real bh s d))
  (#fQ #fKT #fV : perm)
  preserves
    cpu **
    on gpu_loc (gQ |-> Frac fQ sQ ** gKT |-> Frac fKT sKT ** gV |-> Frac fV sV) **
    pure (sQ %~ rQ /\ sKT %~ rKT /\ sV %~ rV)
  requires
    on gpu_loc (gScores |-> sScores0) **
    on gpu_loc (gOut |-> sOut0) **
    pure (
      s * s <= max_blocks * max_threads /\
      SZ.fits (bh * (s * d)) /\
      SZ.fits (bh * (d * s)) /\
      SZ.fits (bh * (s * s)) /\
      bh * s * s <= max_blocks * max_threads /\
      bh * s <= max_blocks /\
      (bh * s) * s <= max_blocks * max_threads /\
      bh * (s * d) <= max_blocks * max_threads
    )
  ensures
    (exists* (probs : chest3 f32 bh s s) (eOut : chest3 f32 bh s d).
      on gpu_loc (gScores |-> probs) **
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ real_sdpa rQ rKT rV))
