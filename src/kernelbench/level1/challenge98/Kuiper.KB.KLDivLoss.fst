module Kuiper.KB.KLDivLoss

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.KLDivLoss
module SZ = Kuiper.SizeT
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module KBMap = Kuiper.KB.Compat.Map
module KS = Kuiper.Seq.Common

let kl_step_approx
  (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (p tt : t) (rp rt : real)
  : Lemma
      (requires p %~ rp /\ tt %~ rt /\ rp >. 0.0R /\ rt >. 0.0R)
      (ensures kl_step #t p tt %~ real_kl_step rp rt)
  = log_approx p rp;
    log_approx tt rt;
    sub_approx (flog tt) (flog p) (log rt) (log rp);
    a_mul tt (sub (flog tt) (flog p)) rt (log rt -. log rp)

#push-options ""
let kl_map_approx
  (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (#n:nat) (sp st : chest1 t n) (rp rt : lseq real n)
  : Lemma
      (requires sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt /\
                positive_seq n rp /\ positive_seq n rt)
      (ensures Map.chest1_map2 (kl_step #t) sp st %~
               seq_to_chest1 (KBMap.lseq_map2 real_kl_step rp rt))
  = let aux (i:natlt n)
      : Lemma
          (acc1 (Map.chest1_map2 (kl_step #t) sp st) i %~
           acc1 (seq_to_chest1 (KBMap.lseq_map2 real_kl_step rp rt)) i)
      = kl_step_approx (acc1 sp i) (acc1 st i)
          (Seq.index rp i) (Seq.index rt i)
    in
    Classical.forall_intro aux
#pop-options

let chest1_seq_roundtrip (#et:Type) (#n:nat) (s : lseq et n)
  : Lemma (Seq.equal (chest1_to_seq (seq_to_chest1 s)) s)
  = ()

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kl_div_loss
  (#t : Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (n : szp {n <= max_blocks * max_threads})
  (batches : szp)
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
    on gpu_loc (predictions |-> sp) **
    pure (positive_seq n rp /\ positive_seq n rt)
  returns res : t
  ensures
    (exists* (sp' : chest1 t n).
       on gpu_loc (predictions |-> sp') **
       pure (res %~ real_kl n batches rp rt))
{
  Map.map_gpu2 (kl_step #t) n predictions targets;
  (* ^ predictions |-> chest1_map2 (kl_step #t) sp st *)

  let terms : erased (lseq real n) =
    hide (KBMap.lseq_map2 real_kl_step rp rt);
  let vr : chest1 real n = hide (seq_to_chest1 (reveal terms));
  kl_map_approx #t (reveal sp) (reveal st) rp rt;
  assert pure (Map.chest1_map2 (kl_step #t) (reveal sp) (reveal st) %~ reveal vr);

  let s = HRed.reduce #t id id 1024sz n predictions vr;
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  chest1_seq_roundtrip (reveal terms);
  assert pure (s %~ real_kl_sum n rp rt);

  let b64 : Int64.t = FStar.Int.Cast.uint64_to_int64
    (FStar.SizeT.sizet_to_uint64 batches);
  assert pure (Int64.v b64 == SZ.v batches);
  let bf : t = of_int b64;
  of_int_approx #t b64;
  assert pure (bf %~ Real.of_int batches);
  let res : t = div s bf;
  div_approx s bf (real_kl_sum n rp rt) (Real.of_int batches);
  res;
}
#pop-options

inline_for_extraction noextract
fn copy1_f32
  (n : szp)
  (src : array1 f32 (l1_forward n) { is_global src })
  (dst : array1 f32 (l1_forward n) { is_global dst })
  (#ss #sd : chest1 f32 n)
  (#f : perm)
  preserves cpu ** on gpu_loc (src |-> Frac f ss)
  requires on gpu_loc (dst |-> sd)
  ensures on gpu_loc (dst |-> ss)
{
  map_loc gpu_loc
    #(dst |-> sd)
    #(core dst |-> to_seq (l1_forward n) sd)
    fn _ { tensor_concr dst; };
  map_loc gpu_loc
    #(src |-> Frac f ss)
    #(core src |-> Frac f (to_seq (l1_forward n) ss))
    fn _ { tensor_concr src; };
  gpu_memcpy_device_to_device (core dst) (core src) n;
  map_loc gpu_loc
    #(core src |-> Frac f (to_seq (l1_forward n) ss))
    #(src |-> Frac f ss)
    fn _ {
      tensor_abs (l1_forward n) (core src);
      rewrite (from_array (l1_forward n) (core src) |-> Frac f ss)
        as (src |-> Frac f ss);
    };
  map_loc gpu_loc
    #(core dst |-> to_seq (l1_forward n) ss)
    #(dst |-> ss)
    fn _ {
      tensor_abs (l1_forward n) (core dst);
      rewrite (from_array (l1_forward n) (core dst) |-> ss)
        as (dst |-> ss);
    }
}

fn kl_scalar_out_f32
  (x : f32)
  preserves cpu
  returns out : array1 f32 (l1_forward 1)
  ensures
    exists* (sout : chest1 f32 1).
      on gpu_loc (out |-> sout) ** pure (acc1 sout 0 == x)
{
  let out = alloc0 #f32 1sz (l1_forward 1);
  Map.map_gpu (fun _ -> x) 1sz out;
  with sout. assert on gpu_loc (out |-> sout);
  assert pure (acc1 sout 0 == x);
  out
}

#push-options "--z3rlimit 40"
fn kl_div_fw_f32
  (n : szp {n <= max_blocks * max_threads})
  (batches : szp)
  (predictions : array1 f32 (l1_forward n) { is_global predictions })
  (targets     : array1 f32 (l1_forward n) { is_global targets })
  (#sp #st : chest1 f32 n)
  (rp rt : erased (lseq real n))
  (#fp #ft : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (predictions |-> Frac fp sp) **
    on gpu_loc (targets |-> Frac ft st) **
    pure (sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt)
  requires
    pure (positive_seq n rp /\ positive_seq n rt)
  returns out : array1 f32 (l1_forward 1)
  ensures
    exists* (sout : chest1 f32 1).
      on gpu_loc (out |-> sout) **
      pure (acc1 sout 0 %~ real_kl n batches rp rt)
{
  let scratch = alloc0 #f32 n (l1_forward n);
  copy1_f32 n predictions scratch;
  let res = kl_div_loss #f32 n batches scratch targets #sp #st rp rt #ft;
  with scratch'. assert on gpu_loc (scratch |-> scratch');
  free scratch;
  let out = kl_scalar_out_f32 res;
  with sout. assert on gpu_loc (out |-> sout);
  assert pure (acc1 sout 0 %~ real_kl n batches rp rt);
  out
}
#pop-options
