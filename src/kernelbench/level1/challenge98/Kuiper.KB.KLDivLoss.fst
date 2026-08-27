module Kuiper.KB.KLDivLoss

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Spec.KLDivLoss
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module KBMap = Kuiper.KB.Compat.Map
module KS = Kuiper.Seq.Common

(* Verified, extractable batchmean reduction: divide the unscaled sum [s] by
   the batch size (extracts to s / (float)(int64_t)(uint64_t)b), so the final
   division is computed inside the verification boundary, not in the bridge. *)
let kl_div_mean_f32 (s : f32) (b : szp) : f32 =
  div s (of_int (FStar.Int.Cast.uint64_to_int64
                   (FStar.SizeT.sizet_to_uint64 b)))

(* The [vr] fed to [reduce] (the [to_real] image of the freshly computed
   [kl_step] chest) is, elementwise, exactly the [to_real_seq] image of
   the seq-level [kl_step] map of the inputs — i.e. the spec's
   [real_kl_sum] integrand.  Both sides are [Seq.init_ghost] of the same
   element function. *)
let kl_vr_eq (#t:Type0) {| scalar t, real_like t, floating t |}
  (#n:nat) (sp st : chest1 t n)
  : Lemma
    (Seq.equal
      (chest1_to_seq (to_real_chest (Map.chest1_map2 (kl_step #t) sp st)))
      (to_real_seq (KBMap.lseq_map2 (kl_step #t) (chest1_to_seq sp) (chest1_to_seq st))))
  = ()

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kl_div_loss
  (#t : Type0) {| scalar t, real_like t, floating t |}
  (n : szp {n <= max_blocks * max_threads})
  (#lp : layout1 n) {| ctlayout lp |}
  (predictions : array1 t lp { is_global predictions })
  (#lt : layout1 n) {| ctlayout lt |}
  (targets     : array1 t lt { is_global targets })
  (#sp : chest1 t n)
  (#st : chest1 t n)
  (#fb : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (targets |-> Frac fb st)
  requires
    on gpu_loc (predictions |-> sp)
  returns res : t
  ensures
    (exists* (sp' : chest1 t n).
       on gpu_loc (predictions |-> sp') **
       pure (res %~ real_kl_sum n (chest1_to_seq sp) (chest1_to_seq st)))
{
  Map.map_gpu2 (kl_step #t) n predictions targets;
  (* ^ predictions |-> chest1_map2 (kl_step #t) sp st *)

  let vr : chest1 real n =
    hide (to_real_chest (Map.chest1_map2 (kl_step #t) (reveal sp) (reveal st)));
  lemma_to_real_chest_approximates
    (Map.chest1_map2 (kl_step #t) (reveal sp) (reveal st));
  assert pure (Map.chest1_map2 (kl_step #t) (reveal sp) (reveal st) %~ reveal vr);

  let s = HRed.reduce #t id id 1024sz n predictions vr;
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  kl_vr_eq #t (reveal sp) (reveal st);
  assert pure (Seq.equal (chest1_to_seq (reveal vr))
                 (to_real_seq (KBMap.lseq_map2 (kl_step #t)
                    (chest1_to_seq (reveal sp)) (chest1_to_seq (reveal st)))));
  assert pure (s %~ real_kl_sum n (chest1_to_seq (reveal sp)) (chest1_to_seq (reveal st)));
  s;
}
#pop-options

let kl_div_fw_f32 : kl_fw_ty f32 =
  fun n predictions targets ->
    kl_div_loss n predictions targets
