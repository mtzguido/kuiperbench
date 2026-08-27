module Kuiper.KB.SDPA

(* KernelBench L1 #97 — Scaled Dot-Product Attention.

     out = softmax(scale * (Q @ K^T)) @ V          (no mask, no dropout)

   Q, K, V have shape (B, H, S, D); the batch+head dims are flattened to
   BH = B*H, giving (BH, S, D) tensors.  K is supplied ALREADY TRANSPOSED
   per page as (BH, D, S) (the host bridge materializes K^T, an unverified
   PyTorch transpose — same pattern as L1 #17).

   This module is a verified ORCHESTRATOR.  It does NOT fuse anything; it
   chains four already-verified Kuiper primitives over the SAME GPU buffers:

     1. batched_gemm_f32 : gScores := Q @ K^T            (exact float)
     2. smul_fw_f32      : gScores *= scale              (exact float)
     3. row_softmax      : gScores := softmax_S(gScores) (the %~ step)
     4. batched_gemm_f32 : gOut    := gScores @ V        (exact float)

   Steps 2 and 3 reinterpret the (BH,S,S) Array3 buffer as a flat Array1
   (for the scalar multiply) resp. a (BH*S, S) Array2 (for the row softmax)
   via pure ghost re-interpretations (no data movement); see the reshape
   lemmas in the .fst, which mirror Kuiper.KB.MatmulND.

   Functional spec (stated at the float level — the two GEMMs and the scalar
   multiply are EXACT; the softmax is the single [%~] step):

     out  = batched_matmul probs V
     probs %~ softmax_pages (to_real (scale * (Q @ K^T)))

   Zero assume * zero magic * zero admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l3_batched_row_major }
module SZ = Kuiper.SizeT
module SM = Kuiper.Spec.Softmax
open Kuiper.KB.BatchedGEMM { batched_matmul }

(* Verified, extractable attention scale 1/sqrt(d) as f32 (see .fst); the
   scale is computed inside the verification boundary. *)
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

inline_for_extraction noextract
type sdpa_ty (t:Type0) {| floating t, real_like t, floating_real_like t |} =
  fn (bh s d : szp)
     (scale : t)
     (gQ  : array3 t (l3_batched_row_major bh s d) { is_global gQ })
     (gKT : array3 t (l3_batched_row_major bh d s) { is_global gKT })
     (gV  : array3 t (l3_batched_row_major bh s d) { is_global gV })
     (gScores : array3 t (l3_batched_row_major bh s s) { is_global gScores })
     (gOut : array3 t (l3_batched_row_major bh s d) { is_global gOut })
     (#sQ  : chest3 t bh s d)
     (#sKT : chest3 t bh d s)
     (#sV  : chest3 t bh s d)
     (#sScores0 : chest3 t bh s s)
     (#sOut0 : chest3 t bh s d)
     (#fQ #fKT #fV : perm)
     requires
       cpu **
       on gpu_loc (gQ |-> Frac fQ sQ ** gKT |-> Frac fKT sKT ** gV |-> Frac fV sV) **
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
       cpu **
       on gpu_loc (gQ |-> Frac fQ sQ ** gKT |-> Frac fKT sKT ** gV |-> Frac fV sV) **
       (exists* (probs : chest3 t bh s s) (eOut : chest3 t bh s d).
         on gpu_loc (gScores |-> probs) **
         on gpu_loc (gOut |-> eOut) **
         pure (probs %~
                 (softmax_pages
                    (to_real_chest
                       (mscale scale (batched_matmul sQ sKT))))) **
         pure (eOut == batched_matmul probs sV))

val sdpa_f32 : sdpa_ty f32
