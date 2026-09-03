module Kuiper.KB.HuberLoss

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.HuberLoss
module SZ = Kuiper.SizeT
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module KBMap = Kuiper.KB.Compat.Map
module KS = Kuiper.Seq.Common

(* Pointwise squared-difference step at the scalar level. *)
// inline_for_extraction
// let huber_step (#t:Type0) {| scalar t, floating t |} (a b : t) : t =
//   let d = sub a b in
//   mul d d

(* Branchless reformulation of the smooth-L1 / Huber step.

   The PyTorch spec branches on |d| < 1, but the [%~] approximation
   relation has NO sound rule connecting a floating comparison to a
   real comparison (near |d| = 1 the fp branch can flip relative to
   the real branch).  So instead of branching we express the step with
   only the operations that DO have approximation rules
   (sub/mul/add/div/fmax/of_int):

     |d|         = fmax d (0 - d)
     max(|d|-1,0)= fmax (|d|-1) 0          (call this [e])
     min(|d|,1)  = |d| - e                 (call this [m])
     smooth_l1   = m*m/2 + e

   This is exact-arithmetic-equal to the branchy spec (see
   [huber_real_eq] below): when |d| < 1, e = 0 and m = |d|, giving
   |d|^2/2; when |d| >= 1, e = |d|-1 and m = 1, giving 1/2 + |d| - 1 =
   |d| - 1/2. *)
inline_for_extraction
let huber_step (#t:Type0) {| scalar t, floating t |} (a b : t) : t =
  let d  = sub a b in
  let ad = fmax d (sub zero d) in        (* |d| *)
  let e  = fmax (sub ad one) zero in     (* max(|d|-1, 0) *)
  let m  = sub ad e in                   (* min(|d|, 1) *)
  add (div (mul m m) (of_int 2L)) e

(* Pure real lemma: the branchless real expression equals the branchy
   PyTorch spec [real_huber_step]. *)
let huber_real_eq (ra rb : real)
  : Lemma (
      (let rd  = ra -. rb in
       let rad = rmax rd (0.0R -. rd) in
       let re  = rmax (rad -. 1.0R) 0.0R in
       let rm  = rad -. re in
       (rm *. rm) /. (Real.of_int 2) +. re)
      == real_huber_step ra rb)
  = let rd  = ra -. rb in
    let rad = rmax rd (0.0R -. rd) in
    assert (rad == rabs rd);             (* rmax rd (-rd) = |rd| *)
    assert (rad *. rad == rd *. rd);     (* (±rd)^2 = rd^2 *)
    assert (Real.of_int 2 == 2.0R);
    ()

let huber_step_approx_lemma (#t:Type0)
  {| scalar t, real_like t, floating t, rr : floating_real_like t |}
  (a b : t)
  (ra rb : real)
  : Lemma (requires a %~ ra /\ b %~ rb)
          (ensures huber_step #t a b %~ real_huber_step ra rb)
          (* The [floating_real_like] instance does not appear in the
             head [%~] pattern, so we add the standard [has_type]
             trigger (same idiom as the *_approx_pat lemmas) to keep
             the SMT pattern well-formed and let it fire per-element
             inside [lseq_map2 ... %~ lseq_map2 ...]. *)
          [SMTPat (huber_step #t a b %~ real_huber_step ra rb);
           SMTPat (has_type rr (floating_real_like t))]
  = of_int_approx #t 2L;
    assert (Real.of_int (Int64.v 2L) == Real.of_int 2);
    assert ((of_int #t 2L) %~ Real.of_int 2);
    (* Build the approximations step by step; the *_approx_pat SMTPats
       (sub/fmax/mul/div/add) and a0/a1 fire on these terms. *)
    let rd  = ra -. rb in
    let d : t = sub a b in
    assert (d %~ rd);
    let nd : t = sub zero d in
    assert (nd %~ (0.0R -. rd));
    let rad = rmax rd (0.0R -. rd) in
    let ad : t = fmax d nd in
    assert (ad %~ rad);
    let re  = rmax (rad -. 1.0R) 0.0R in
    let e : t = fmax (sub ad one) zero in
    assert (e %~ re);
    let rm  = rad -. re in
    let m : t = sub ad e in
    assert (m %~ rm);
    assert (mul m m %~ (rm *. rm));
    assert (Real.of_int 2 =!= 0.0R);
    assert (div (mul m m) (of_int 2L) %~ ((rm *. rm) /. Real.of_int 2));
    assert (add (div (mul m m) (of_int 2L)) e
              %~ (((rm *. rm) /. Real.of_int 2) +. re));
    huber_real_eq ra rb

let seq_map_id_eq (#a:Type) (s : Seq.seq a)
  : Lemma (Seq.equal (KS.seq_map id s) s)
  = ()

(* Elementwise lift: the f32 branchless-Huber map of the two GPU
   ownership chests approximates the real Huber-step map of [rp]/[rt]. *)
#push-options ""
let huber_map_approx (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (#n:nat) (sp st : chest1 t n) (rp rt : lseq real n)
  : Lemma (requires sp %~ seq_to_chest1 rp /\ st %~ seq_to_chest1 rt)
          (ensures Map.chest1_map2 huber_step sp st
                   %~ seq_to_chest1 (KBMap.lseq_map2 real_huber_step rp rt))
  = let aux (i:natlt n)
      : Lemma (acc1 (Map.chest1_map2 huber_step sp st) i
               %~ acc1 (seq_to_chest1 (KBMap.lseq_map2 real_huber_step rp rt)) i)
      = huber_step_approx_lemma (acc1 sp i) (acc1 st i)
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
fn huber_loss
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
       pure (res %~ real_huber n rp rt))
{
  Map.map_gpu2 huber_step n predictions targets;
  (* ^ This clobbers predictions with [chest1_map2 huber_step sp st]. *)

  let vr : chest1 real n =
    hide (seq_to_chest1 (KBMap.lseq_map2 real_huber_step rp rt));
  huber_map_approx #t (reveal sp) (reveal st) (reveal rp) (reveal rt);
  assert pure (Map.chest1_map2 huber_step (reveal sp) (reveal st) %~ reveal vr);

  let s = HRed.reduce #t id id 1024sz n predictions vr;
  assert pure (equal (chest_map id (reveal vr)) (reveal vr));
  chest1_seq_roundtrip (KBMap.lseq_map2 real_huber_step rp rt);
  assert pure (Seq.equal (chest1_to_seq (reveal vr))
                         (KBMap.lseq_map2 real_huber_step rp rt));
  assert pure (s %~ rsum (KBMap.lseq_map2 real_huber_step rp rt));

  let n64 : Int64.t = FStar.Int.Cast.uint64_to_int64 (FStar.SizeT.sizet_to_uint64 n);
  assert pure (Int64.v n64 == SZ.v n);
  let nn : t = of_int n64;
  of_int_approx #t n64;
  assert pure (nn %~ Real.of_int n);

  let res : t = div s nn;
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

fn huber_scalar_out_f32
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
fn huber_loss_fw_f32
  (n : szp {n <= max_blocks * max_threads})
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
  returns out : array1 f32 (l1_forward 1)
  ensures
    exists* (sout : chest1 f32 1).
      on gpu_loc (out |-> sout) **
      pure (acc1 sout 0 %~ real_huber n rp rt)
{
  let scratch = alloc0 #f32 n (l1_forward n);
  copy1_f32 n predictions scratch;
  let res = huber_loss #f32 n scratch targets #sp #st rp rt #ft;
  with scratch'. assert on gpu_loc (scratch |-> scratch');
  free scratch;
  let out = huber_scalar_out_f32 res;
  with sout. assert on gpu_loc (out |-> sout);
  assert pure (acc1 sout 0 %~ real_huber n rp rt);
  out
}
#pop-options
