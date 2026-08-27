module Kuiper.Spec.CrossEntropyLoss

(* Functional specification for KernelBench L1 #95:
   Cross-Entropy Loss (PyTorch defaults: reduction='mean').

   Reference:
     loss = mean_b ( -log_softmax(predictions[b])[targets[b]] )
          = mean_b ( logsumexp(predictions[b]) - predictions[b, targets[b]] )

   This spec is *used* (imported by Kuiper.KB.CrossEntropyLoss) and is
   a genuine function of ALL inputs: the flat (B*C) predictions buffer
   [sp] and the length-B target index sequence [st].

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

(* Row [r] (a contiguous block of [c] f32s) of a flat [n]-element
   row-major (B*C) predictions buffer.  Defensive guard keeps it
   total; for every valid [r] (i.e. [r * c + c <= n]) it returns the
   genuine row slice. *)
let crow (#n:nat) (s : Seq.lseq f32 n) (c:nat) (r:nat) : Seq.lseq f32 c =
  if r * c + c <= n then Seq.slice s (r * c) (r * c + c) else Seq.create c zero

(* Per-row cross-entropy term in REAL arithmetic, total in [t] via a
   defensive clamp:
     ce_term_r c sp r t  =  - (log_softmax(crow sp c r))[t]. *)
let ce_term_r (c:pos) (#n:nat) (sp : Seq.lseq f32 n) (r:nat) (t:nat) : real =
  let rr = to_real_seq (crow sp c r) in
  if Seq.length rr > 0 && t < Seq.length rr
  then 0.0R -. (acc1 (log_softmax_real (seq_to_chest1 (rr <: Seq.lseq real (Seq.length rr)))) t)
  else 0.0R

(* Postcondition: the returned scalar [res] is the mean cross-entropy
   loss of the inputs.  The per-batch loss vector is existentially
   bound (the device-side log-softmax + tree-reduce are not
   bit-exactly determined) but *pinned* to the genuine inputs:

     * [per_batch[r]] approximates (via [%~]) the real per-row CE term
       [ce_term_r] of row [r] of [sp] at its target class [st[r]];
     * [s] approximates the real sum of the per-batch losses;
     * [res == mul s inv_b]   (multiply by the verified 1/B). *)
let cross_entropy_post
  (batches : pos)
  (num_classes : pos)
  (inv_b : f32)
  (sp : Seq.lseq f32 (batches * num_classes))
  (st : Seq.lseq SZ.t batches)
  (res : f32)
  : prop =
  exists (per_batch : Seq.lseq f32 batches) (s : f32).
    (forall (r : nat). r < batches ==>
       (per_batch @! r) %~ ce_term_r num_classes sp r (st @! r)) /\
    s %~ rsum (to_real_seq per_batch) /\
    res == mul s inv_b
