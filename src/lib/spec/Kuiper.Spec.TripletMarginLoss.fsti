module Kuiper.Spec.TripletMarginLoss

(* Functional specification for KernelBench L1 #99:
   Triplet Margin Loss (PyTorch defaults p=2, eps=1e-6,
   reduction='mean').

   Reference:
     dist_eps(x, y) = ||x - y + eps * 1||_2
     loss = mean_b max(0, dist_eps(a_b, p_b) - dist_eps(a_b, n_b) + margin)

   The spec composes:
     * per-row squared Euclidean distance, pinned in *real* arithmetic
       via the [%~] approximation relation to the genuine inputs
       [sa] / [sp] / [sn]:
         sumsq_ap[r] %~ sum_j (a[r,j] - p[r,j] + eps)^2    (reals)
       (we keep the f32 [sqrt] opaque: [d_ap[r] == sqrt sumsq_ap[r]],
        since there is no real-valued [sqrt] approximation lemma);
     * the elementwise [triplet_step] over the two length-B distance
       vectors;
     * the mean reduction, pinned via [%~] against the real sum.

   This makes the spec genuinely about triplet-margin loss rather than
   "any output": the existentially bound per-row [sumsq] vectors are
   forced (up to f32 rounding, via [%~]) to be the true real-valued
   squared distances of the corresponding rows of [sa] / [sp] / [sn],
   and the returned scalar is forced to be their mean
   triplet-margin-loss.  No admits / assumes / magic. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Seq.Common { (@!) }
open Kuiper.Float32
module Seq = FStar.Seq
unfold let f32 = Kuiper.Float32.t

(* Per-batch margin step: max(0, d_ap - d_an + margin). *)
inline_for_extraction
let triplet_step (margin : f32) (d_ap d_an : f32) : f32 =
  fmax zero (add (sub d_ap d_an) margin)

(* Real-arithmetic squared-difference step: the genuine functional
   behaviour that the kernel's f32 squared-difference step
   approximates. *)
let sqdiff_step_r (eps : f32) (ra rp : real) : real =
  let d = (ra -. rp) +. to_real eps in d *. d

(* Total row accessor: extract row [r] (a contiguous block of [d]
   f32s) from a flat [n]-element sequence.  The defensive guard keeps
   it total; for every valid [r] (i.e. [r * d + d <= n]) it returns
   the genuine row slice. *)
let trow (#n:nat) (s : Seq.lseq f32 n) (d:nat) (r:nat) : Seq.lseq f32 d =
  if r * d + d <= n then Seq.slice s (r * d) (r * d + d) else Seq.create d zero

(* TRUE real squared Euclidean distance between two f32 rows: the sum
   over the [d] coordinates of the squared real differences. *)
let real_sq_dist (eps : f32) (d:nat) (ra rb : Seq.lseq f32 d) : real =
  rsum (Seq.init d (fun j ->
    sqdiff_step_r eps (to_real (ra @! j)) (to_real (rb @! j))))

(* Postcondition: the returned scalar [res] is the mean
   triplet-margin-loss of the input rows.  All distance vectors are
   existentially bound (the device-side tree-reduce and the f32
   [sqrt] are not bit-exactly determined), but they are *pinned* to
   the genuine inputs:

     * [sumsq_ap[r]] / [sumsq_an[r]] approximate (via [%~]) the true
       real squared distances of row [r] of [sa] against [sp] / [sn];
     * [d_ap[r] == sqrt sumsq_ap[r]] and [d_an[r] == sqrt sumsq_an[r]]
       (exact f32 equations -- [sqrt] is opaque to the real model);
     * [s] approximates the real sum of the per-row [triplet_step]s;
     * [res == mul s inv_b]. *)
let triplet_post
  (batches : pos)
  (d : nat)
  (margin : f32)
  (eps : f32)
  (inv_b : f32)
  (sa sp sn : Seq.lseq f32 (batches * d))
  (res : f32)
  : prop =
  exists (sumsq_ap sumsq_an d_ap d_an : Seq.lseq f32 batches) (s : f32).
    (forall (r : nat). r < batches ==>
       (d_ap @! r) == sqrt (sumsq_ap @! r) /\
       (d_an @! r) == sqrt (sumsq_an @! r) /\
       (sumsq_ap @! r) %~ real_sq_dist eps d (trow sa d r) (trow sp d r) /\
       (sumsq_an @! r) %~ real_sq_dist eps d (trow sa d r) (trow sn d r)) /\
    s %~ rsum (to_real_seq (Seq.init batches (fun r ->
      triplet_step margin (d_ap @! r) (d_an @! r)))) /\
    res == mul s inv_b
