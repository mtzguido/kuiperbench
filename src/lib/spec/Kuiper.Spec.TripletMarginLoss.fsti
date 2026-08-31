module Kuiper.Spec.TripletMarginLoss

(* Functional specification for KernelBench L1 #99:
   Triplet Margin Loss (PyTorch defaults p=2, eps=1e-6,
   reduction='mean').

   The public postcondition is directly real-valued.  For real sequences
   approximated by the three f32 inputs it states:

     dist_eps(x, y) = sqrt (sum_j (x_j - y_j + eps)^2)
     loss = mean_b max(0, dist_eps(a_b, p_b)
                          - dist_eps(a_b, n_b) + margin)

   There are no existential floating-point intermediates in [triplet_post].
   The proof crosses each f32 square root through [sqrt_approx] and composes
   the ordinary arithmetic approximation laws through the margin step and
   final mean. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Approximates
open Kuiper.Seq.Common { (@!) }
module Seq = FStar.Seq
module RealSqrt = FStar.Math.Sqrt
unfold let f32 = Kuiper.Float32.t

(* Total row accessor for a flat real-valued BxD input. *)
let trow (#n:nat) (s : Seq.lseq real n) (d:nat) (r:nat)
  : Seq.lseq real d =
  if r * d + d <= n
  then Seq.slice s (r * d) (r * d + d)
  else Seq.create d 0.0R

let sqdiff_step_r (eps ra rb : real) : real =
  let delta = (ra -. rb) +. eps in
  delta *. delta

let real_sq_dist
  (eps : real)
  (d : nat)
  (ra rb : Seq.lseq real d)
  : real =
  rsum (Seq.init d (fun j -> sqdiff_step_r eps (ra @! j) (rb @! j)))

val real_sq_dist_nonnegative
  (eps : real)
  (d : nat)
  (ra rb : Seq.lseq real d)
  : Lemma (real_sq_dist eps d ra rb >=. 0.0R)

let real_dist
  (eps : real)
  (d : nat)
  (ra rb : Seq.lseq real d)
  : RealSqrt.rnonneg =
  let sq = real_sq_dist eps d ra rb in
  real_sq_dist_nonnegative eps d ra rb;
  sq

let real_triplet_step (margin d_ap d_an : real) : real =
  rmax 0.0R ((d_ap -. d_an) +. margin)

let real_triplet_terms
  (batches : pos)
  (d : nat)
  (margin eps : real)
  (ra rp rn : Seq.lseq real (batches * d))
  : Seq.lseq real batches =
  Seq.init batches (fun r ->
    real_triplet_step margin
      (RealSqrt.sqrt (real_dist eps d (trow ra d r) (trow rp d r)))
      (RealSqrt.sqrt (real_dist eps d (trow ra d r) (trow rn d r))))

let real_triplet_loss
  (batches : pos)
  (d : nat)
  (margin eps : real)
  (ra rp rn : Seq.lseq real (batches * d))
  : real =
  rsum (real_triplet_terms batches d margin eps ra rp rn)
    /. FStar.Real.of_int batches

val real_triplet_loss_mul
  (batches : pos)
  (d : nat)
  (margin eps : real)
  (ra rp rn : Seq.lseq real (batches * d))
  : Lemma
      (real_triplet_loss batches d margin eps ra rp rn ==
       rsum (real_triplet_terms batches d margin eps ra rp rn) *.
         (1.0R /. FStar.Real.of_int batches))

let triplet_post
  (batches : pos)
  (d : nat)
  (margin eps : f32)
  (ra rp rn : Seq.lseq real (batches * d))
  (res : f32)
  : prop =
  res %~ real_triplet_loss batches d (to_real margin) (to_real eps) ra rp rn
