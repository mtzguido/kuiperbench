module Kuiper.KB.MSELoss

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.MSELoss
module SZ = Kuiper.SizeT
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module KBMap = Kuiper.KB.Compat.Map
module KS = Kuiper.Seq.Common

(* Pointwise squared-difference step at the scalar level. *)
inline_for_extraction
let mse_step (#t:Type0) {| scalar t, floating t |} (a b : t) : t =
  let d = sub a b in
  mul d d

let mse_step_approx_lemma (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (a b : t)
  (ra rb : real)
  : Lemma (requires a %~ ra /\ b %~ rb)
          (ensures mse_step #t a b %~ real_mse_step ra rb)
  = ()

(* Elementwise lift: the f32 squared-difference map of the two GPU
   ownership chests approximates the real squared-difference map of the
   spec [rp]/[rt].  Proved per-element in a small pure context (mirrors
   [Kuiper.Kernel.LogSoftmax.log_softmax_approx]). *)
#push-options ""
let mse_map_approx (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (#n:nat) (sp st : chest1 t n) (rp rt : lseq real n)
  : Lemma (requires sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt)
          (ensures Map.chest1_map2 mse_step sp st
                   %~ seq_to_chest1 (KBMap.lseq_map2 real_mse_step rp rt))
  = let aux (i:natlt n)
      : Lemma (acc1 (Map.chest1_map2 mse_step sp st) i
               %~ acc1 (seq_to_chest1 (KBMap.lseq_map2 real_mse_step rp rt)) i)
      = mse_step_approx_lemma (acc1 sp i) (acc1 st i)
          (Seq.index rp i) (Seq.index rt i)
    in
    Classical.forall_intro aux
#pop-options

(* [chest1_to_seq] is a left inverse of [seq_to_chest1]. *)
let chest1_seq_roundtrip (#et:Type) (#n:nat) (s : lseq et n)
  : Lemma (Seq.equal (chest1_to_seq (seq_to_chest1 s)) s)
  = ()

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn mse_loss
  (#t : Type0) {| scalar t, real_like t, floating t, floating_real_like t |}
  (n : szp {n <= max_blocks * max_threads})
  (#lp : layout1 n) {| ctlayout lp |}
  (predictions : array1 t lp { is_global predictions })
  (#lt : layout1 n) {| ctlayout lt |}
  (targets     : array1 t lt { is_global targets })
  (#sp : chest1 t n)
  (#st : chest1 t n)
  (rp : erased (lseq real n))
  (rt : erased (lseq real n))
  (#fb : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (targets |-> Frac fb st) **
    pure (sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt)
  requires
    on gpu_loc (predictions |-> sp)
  returns res : t
  ensures
    (exists* (sp' : chest1 t n).
       on gpu_loc (predictions |-> sp') **
       pure (res %~ real_mse n rp rt))
{
  Map.map_gpu2 mse_step n predictions targets;
  (* ^ This clobbers predictions with [chest1_map2 mse_step sp st]. *)

  let vr : chest1 real n =
    hide (seq_to_chest1 (KBMap.lseq_map2 real_mse_step rp rt));
  mse_map_approx #t (reveal sp) (reveal st) (reveal rp) (reveal rt);
  assert pure (Map.chest1_map2 mse_step (reveal sp) (reveal st) %~ reveal vr);

  let s = HRed.reduce #t id id 1024sz n predictions vr;
  (* reduce ⇒ s %~ rsum (chest1_to_seq (chest_map id vr)) *)
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  chest1_seq_roundtrip (KBMap.lseq_map2 real_mse_step rp rt);
  assert pure (Seq.equal (chest1_to_seq (reveal vr))
                         (KBMap.lseq_map2 real_mse_step rp rt));
  assert pure (s %~ rsum (KBMap.lseq_map2 real_mse_step rp rt));

  let n64 : Int64.t = FStar.Int.Cast.uint64_to_int64 (FStar.SizeT.sizet_to_uint64 n);
  assert pure (Int64.v n64 == SZ.v n);
  let nn : t = of_int n64;
  of_int_approx #t n64;
  assert pure (nn %~ Real.of_int n);

  let res : t = div s nn;
  res;
}
#pop-options

let mse_loss_fw_f32 : mse_fw_ty f32 =
  fun n predictions targets ->
    mse_loss n predictions targets
