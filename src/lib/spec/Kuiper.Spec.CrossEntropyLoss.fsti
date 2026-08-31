module Kuiper.Spec.CrossEntropyLoss

(* Functional specification for KernelBench L1 #95:
   Cross-Entropy Loss (PyTorch defaults: reduction='mean').

   Reference:
     loss = mean_b ( -log_softmax(predictions[b])[targets[b]] )
          = mean_b ( logsumexp(predictions[b]) - predictions[b, targets[b]] )

   The public postcondition is directly real-valued.  For a real logits
   sequence [rp] approximated by the f32 predictions, it states that the
   result approximates the mean of the mathematical real log-softmax
   losses selected by the (exact integer) targets.

   The per-row term is pinned to the verified
   [Kuiper.Kernel.LogSoftmax.log_softmax_real] path that the kernel
   actually computes (numerically-stable subtract-max log-softmax):

     ce_term_r c sp r t  ==  - (log_softmax(row_r))[t]

   which is mathematically equal to  logsumexp(row_r) - row_r[t], the
   PyTorch per-sample cross entropy.  No admits / assumes / magic. *)

open Kuiper.Common
open Kuiper.Chest
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Seq.Common { (@!) }
open Kuiper.Float32
open Kuiper.Kernel.LogSoftmax { log_softmax_real }
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
unfold let f32 = Kuiper.Float32.t

(* Row [r] (a contiguous block of [c] reals) of a flat [n]-element
   row-major (B*C) predictions buffer.  Defensive guard keeps it
   total; for every valid [r] (i.e. [r * c + c <= n]) it returns the
   genuine row slice. *)
let crow (#n:nat) (s : Seq.lseq real n) (c:nat) (r:nat) : Seq.lseq real c =
  if r * c + c <= n then Seq.slice s (r * c) (r * c + c) else Seq.create c 0.0R

(* Per-row cross-entropy term in REAL arithmetic, total in [t] via a
   defensive clamp:
     ce_term_r c sp r t  =  - (log_softmax(crow sp c r))[t]. *)
let ce_term_r (c:pos) (#n:nat) (rp : Seq.lseq real n) (r:nat) (t:nat) : real =
  let rr = crow rp c r in
  if t < c
  then 0.0R -. (acc1 (log_softmax_real (seq_to_chest1 rr)) t)
  else 0.0R

let real_cross_entropy_terms
  (batches : pos)
  (num_classes : pos)
  (rp : Seq.lseq real (batches * num_classes))
  (st : Seq.lseq SZ.t batches)
  : Seq.lseq real batches =
  Seq.init batches (fun r -> ce_term_r num_classes rp r (SZ.v (st @! r)))

let real_cross_entropy
  (batches : pos)
  (num_classes : pos)
  (rp : Seq.lseq real (batches * num_classes))
  (st : Seq.lseq SZ.t batches)
  : real =
  rsum (real_cross_entropy_terms batches num_classes rp st)
    /. FStar.Real.of_int batches

val real_cross_entropy_mul
  (batches : pos)
  (num_classes : pos)
  (rp : Seq.lseq real (batches * num_classes))
  (st : Seq.lseq SZ.t batches)
  : Lemma
      (real_cross_entropy batches num_classes rp st ==
       rsum (real_cross_entropy_terms batches num_classes rp st) *.
         (1.0R /. FStar.Real.of_int batches))

(* No floating-point intermediate is existentially chosen here. *)
let cross_entropy_post
  (batches : pos)
  (num_classes : pos)
  (rp : Seq.lseq real (batches * num_classes))
  (st : Seq.lseq SZ.t batches)
  (res : f32)
  : prop =
  res %~ real_cross_entropy batches num_classes rp st
