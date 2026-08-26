module Kuiper.KB.CrossEntropyLoss

(* KernelBench L1 #95: Cross-Entropy Loss (mean reduction).

   Reference (PyTorch torch.nn.functional.cross_entropy, defaults):
       loss = mean_b ( -log_softmax(predictions[b])[targets[b]] )
            = mean_b ( logsumexp(predictions[b]) - predictions[b, targets[b]] )

   This module exposes a single self-contained verified entry point
   [ce_loss_fw_f32] that performs the WHOLE computation inside the
   verification boundary:

     * for each batch row [b] it copies the length-C row into scratch,
       runs the verified numerically-stable [log_softmax_gpu] in place,
       gathers the (negated) target lane -- the per-row CE loss;
     * it writes all B per-row losses into a device buffer and
       reduce-sums them on-device with the verified [HReduce.reduce];
     * it multiplies the sum by the verified reciprocal [1/B].

   The returned scalar is therefore VERIFIED as a function of ALL
   inputs (predictions + targets): its post is
   [Kuiper.Spec.CrossEntropyLoss.cross_entropy_post].  The C bridge does
   a single call -- no host reduction loop.

   This mirrors the per-row-then-batched-reduce pattern of the verified
   L1 #99 TripletMarginLoss kernel. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.CrossEntropyLoss
module SZ = Kuiper.SizeT

(* Verified, extractable reciprocal 1/B as f32 (see .fst). *)
val ce_recip_f32 (b : szp) : f32

inline_for_extraction noextract
type ce_loss_fw_ty =
  fn
    (b : szp { b <= max_blocks * max_threads /\
               SZ.fits (b + max_threads) })
    (c : szp { c <= max_blocks * max_threads /\
               SZ.fits (c + max_threads) /\
               SZ.fits (b * c) })
    (inv_b : f32)
    (predictions : array1 f32 (l1_forward (b *^ c)) { is_global predictions })
    (targets : array1 SZ.t (l1_forward b) { is_global targets })
    (#sp : erased (chest1 f32 (b *^ c)))
    (#stv : erased (chest1 SZ.t b))
    (#fp #ft : perm)
    preserves cpu **
              on gpu_loc (predictions |-> Frac fp sp) **
              on gpu_loc (targets |-> Frac ft stv)
    requires
      pure (forall (r : nat). r < SZ.v b ==> SZ.v (acc1 (reveal stv) r) < SZ.v c)
    returns res : f32
    ensures
      pure (cross_entropy_post (SZ.v b) (SZ.v c) inv_b
              (chest1_to_seq (reveal sp) <: Seq.lseq f32 (SZ.v b * SZ.v c))
              (chest1_to_seq (reveal stv) <: Seq.lseq SZ.t (SZ.v b))
              res)

val ce_loss_fw_f32 : ce_loss_fw_ty
