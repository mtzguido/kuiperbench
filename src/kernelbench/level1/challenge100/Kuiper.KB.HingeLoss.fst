module Kuiper.KB.HingeLoss

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Spec.HingeLoss
module SZ = Kuiper.SizeT
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module KBMap = Kuiper.KB.Compat.Map

(* Pointwise hinge step at the scalar (e.g. f32) level. *)
inline_for_extraction
let hinge_step (#t:Type0) {| scalar t, floating t |} (p target : t) : t =
  fmax zero (sub one (mul p target))

let hinge_step_approx_lemma (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (p target : t)
  (rp rt : real)
  : Lemma (requires p %~ rp /\ target %~ rt)
          (ensures hinge_step #t p target %~ real_hinge_step rp rt)
  = ()

(* Elementwise lift: the f32 hinge-step map of the two GPU ownership
   chests approximates the real hinge-step map of [rp]/[rt]. *)
#push-options ""
let hinge_map_approx (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (#n:nat) (sp st : chest1 t n) (rp rt : lseq real n)
  : Lemma (requires sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt)
          (ensures Map.chest1_map2 hinge_step sp st
                   %~ seq_to_chest1 (KBMap.lseq_map2 real_hinge_step rp rt))
  = let aux (i:natlt n)
      : Lemma (acc1 (Map.chest1_map2 hinge_step sp st) i
               %~ acc1 (seq_to_chest1 (KBMap.lseq_map2 real_hinge_step rp rt)) i)
      = hinge_step_approx_lemma (acc1 sp i) (acc1 st i)
          (Seq.index rp i) (Seq.index rt i)
    in
    Classical.forall_intro aux
#pop-options

(* [chest1_to_seq] is a left inverse of [seq_to_chest1]. *)
let chest1_seq_roundtrip (#et:Type) (#n:nat) (s : lseq et n)
  : Lemma (Seq.equal (chest1_to_seq (seq_to_chest1 s)) s)
  = ()

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn hinge_loss
  (#t : Type0) {| scalar t, real_like t, floating t, floating_real_like t |}
  (n : szp {n <= max_blocks * max_threads})
  (#lp : layout1 n) {| ctlayout lp |}
  (predictions : array1 t lp { is_global predictions })
  (#lt : layout1 n) {| ctlayout lt |}
  (targets     : array1 t lt { is_global targets })
  (#sp : erased (chest1 t n))
  (#st : erased (chest1 t n))
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
       pure (res %~ real_hinge n rp rt))
{
  Map.map_gpu2 hinge_step n predictions targets;

  let vr : erased (chest1 real n) =
    hide (seq_to_chest1 (KBMap.lseq_map2 real_hinge_step rp rt));
  hinge_map_approx #t (reveal sp) (reveal st) (reveal rp) (reveal rt);
  assert pure (Map.chest1_map2 hinge_step (reveal sp) (reveal st) %~ reveal vr);

  let s = HRed.reduce #t id id 1024sz n predictions vr;
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  chest1_seq_roundtrip (KBMap.lseq_map2 real_hinge_step rp rt);
  assert pure (Seq.equal (chest1_to_seq (reveal vr))
                         (KBMap.lseq_map2 real_hinge_step rp rt));
  assert pure (s %~ rsum (KBMap.lseq_map2 real_hinge_step rp rt));

  let n64 : Int64.t = FStar.Int.Cast.uint64_to_int64 (FStar.SizeT.sizet_to_uint64 n);
  assert pure (Int64.v n64 == SZ.v n);
  let nn : t = of_int n64;
  of_int_approx #t n64;
  assert pure (nn %~ Real.of_int n);

  let res : t = div s nn;
  res;
}
#pop-options

let hinge_loss_fw_f32 : hinge_fw_ty f32 =
  fun n predictions targets rp rt ->
    hinge_loss n predictions targets rp rt
