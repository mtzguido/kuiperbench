module Kuiper.KB.SeparableConv2D

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Spec.DepthwiseConv2D
open Kuiper.Spec.PointwiseConv2D
open Kuiper.Spec.SeparableConv2D
open Kuiper.Kernel.Conv2D.Naive
open Kuiper.Kernel.Conv2D.Depthwise
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas
module DW = Kuiper.KB.DepthwiseConv2D
module CG = Kuiper.KB.Conv2DGeneral

let separable_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)
  = DW.dwconv2d_out_dim n k stride pad

(* ================================================================== *)
(* Pure composition lemmas (ADMIT-free).                              *)
(* ================================================================== *)

(* Mixed-radix index round-trip: decoding the row-major flattening of   *)
(* [(bi, ci, oh, ow)] over [(b, c, h, w)] recovers the four indices.    *)
let decode4_lemma (b c h w : pos)
  (bi : natlt b) (ci : natlt c) (oh : natlt h) (ow : natlt w)
  : Lemma
    (let chw = c * h * w in
     let hw = h * w in
     let tid = ((bi * c + ci) * h + oh) * w + ow in
     tid < b * chw /\
     tid / chw == bi /\
     (tid % chw) / hw == ci /\
     ((tid % chw) % hw) / w == oh /\
     ((tid % chw) % hw) % w == ow)
  = let hw = h * w in
    let chw = c * h * w in
    let inner_hw = oh * w + ow in
    let inner_chw = ci * hw + inner_hw in
    (* chw == c * hw *)
    ML.paren_mul_right c h w;
    assert (chw == c * hw);
    (* bounds *)
    ML.lemma_mult_le_right w oh (h - 1);
    ML.distributivity_sub_left h 1 w;
    assert (inner_hw < hw);
    ML.lemma_mult_le_right hw ci (c - 1);
    ML.distributivity_sub_left c 1 hw;
    assert (inner_chw < chw);
    (* algebra: tid == inner_chw + bi*chw *)
    let tid = ((bi * c + ci) * h + oh) * w + ow in
    ML.distributivity_add_left ((bi * c + ci) * h) oh w;
    ML.paren_mul_right (bi * c + ci) h w;
    ML.distributivity_add_left (bi * c) ci hw;
    ML.paren_mul_right bi c hw;
    assert (tid == inner_chw + bi * chw);
    (* outermost div/mod by chw *)
    ML.lemma_div_plus inner_chw bi chw;
    ML.small_div inner_chw chw;
    ML.lemma_mod_plus inner_chw bi chw;
    ML.small_mod inner_chw chw;
    (* middle div/mod by hw *)
    ML.lemma_div_plus inner_hw ci hw;
    ML.small_div inner_hw hw;
    ML.lemma_mod_plus inner_hw ci hw;
    ML.small_mod inner_hw hw;
    (* innermost div/mod by w *)
    ML.lemma_div_plus ow oh w;
    ML.small_div ow w;
    ML.lemma_mod_plus ow oh w;
    ML.small_mod ow w

(* Step A: the depthwise-output flat seq, viewed as an [etensor4], IS the   *)
(* spec [dwconv2d] tensor (extensional equality from the per-cell post).    *)
let mid_eq_lemma
  (#et:Type) {| scalar et |}
  (b c : pos) (h_in w_in : nat) (kh kw : pos) (stride : pos) (pad : nat)
  (h_out w_out : pos)
  (sx : chest1 et (b*c*h_in*w_in))
  (sw_dw : chest1 et (c*1*kh*kw))
  (sbias_dw : chest1 et c)
  (smid : chest1 et (b*c*h_out*w_out))
  : Lemma
    (requires (forall (t':nat{t' < b*c*h_out*w_out}).
                 acc1 smid t' ==
                 dwconv2d_out_at b c h_in w_in kh kw stride pad
                                 h_out w_out sx sw_dw sbias_dw t'))
    (ensures
       lseq_to_t4 b c h_out w_out smid ==
       dwconv2d kh kw stride stride pad pad 1 1 h_out w_out
         (lseq_to_t4 b c h_in w_in sx)
         (lseq_to_t4 c 1 kh kw sw_dw)
         (chest1_to_seq sbias_dw))
  = let xt = lseq_to_t4 b c h_in w_in sx in
    let wt = lseq_to_t4 c 1 kh kw sw_dw in
    let dwt = dwconv2d kh kw stride stride pad pad 1 1 h_out w_out xt wt (chest1_to_seq sbias_dw) in
    let midt = lseq_to_t4 b c h_out w_out smid in
    introduce forall (bi:natlt b) (ci:natlt c) (oh:natlt h_out) (ow:natlt w_out).
      tacc midt bi ci oh ow == tacc dwt bi ci oh ow
    with (
      decode4_lemma b c h_out w_out bi ci oh ow;
      lemma_dwconv2d_index kh kw stride stride pad pad 1 1 h_out w_out
        xt wt (chest1_to_seq sbias_dw) bi ci oh ow
    );
    lemma_t4_equal_intro midt dwt;
    etensor4_ext midt dwt

(* Step B (recursion): partial 1x1-conv accumulator == pointwise accumulator. *)
let rec pw_eq_aux
  (#et:Type) {| scalar et |}
  (b : nat) (c : pos) (h w : nat) (cout : nat)
  (mid : etensor4 et b c h w)
  (pw4 : etensor4 et cout c 1 1)
  (bi : natlt b) (oc : natlt cout) (oh : natlt h) (ow : natlt w)
  (to : nat{to <= c})
  : Lemma
    (ensures
      __conv2d_single 1 1 1 0 mid pw4 bi oc oh ow to ==
      __pwconv2d_single mid
        (mk2 (fun (oc:natlt cout) (ic:natlt c) -> tacc pw4 oc ic 0 0))
        bi oc oh ow to)
    (decreases to)
  = let mat : chest2 et cout c =
      mk2 (fun (oc:natlt cout) (ic:natlt c) -> tacc pw4 oc ic 0 0) in
    if to = 0 then (
      __conv2d_single_zero_lemma 1 1 1 0 mid pw4 bi oc oh ow;
      __pwconv2d_single_zero_lemma mid mat bi oc oh ow
    ) else (
      pw_eq_aux b c h w cout mid pw4 bi oc oh ow (to - 1);
      __conv2d_single_lemma c 1 1 1 0 mid pw4 bi oc oh ow to;
      __pwconv2d_single_lemma mid mat bi oc oh ow to;
      let i = to - 1 in
      assert (unrank_ic c 1 1 i == i);
      assert (unrank_kh c 1 1 i == 0);
      assert (unrank_kw c 1 1 i == 0);
      assert (acc2 mat oc i == tacc pw4 oc i 0 0)
    )

(* Step B (full): 1x1 general-conv per-pixel == pointwise per-pixel.        *)
let pw_eq_lemma
  (#et:Type) {| scalar et |}
  (b : nat) (c : pos) (h w : nat) (cout : nat)
  (mid : etensor4 et b c h w)
  (pw4 : etensor4 et cout c 1 1)
  (bias : Seq.lseq et cout)
  (bi : natlt b) (oc : natlt cout) (oh : natlt h) (ow : natlt w)
  : Lemma
    (ensures
      conv2d_single 1 1 1 0 mid pw4 bias bi oc oh ow ==
      pwconv2d_single mid
        (mk2 (fun (oc:natlt cout) (ic:natlt c) -> tacc pw4 oc ic 0 0))
        bias bi oc oh ow)
  = pw_eq_aux b c h w cout mid pw4 bi oc oh ow c

(* Composition: each pointwise output cell equals the whole separable spec. *)
let separable_compose_lemma
  (#et:Type) {| scalar et |}
  (b c : pos) (h_in w_in : nat) (kh kw : pos) (stride : pos) (pad : nat)
  (cout : pos) (h_out w_out : pos)
  (sx : chest1 et (b*c*h_in*w_in))
  (sw_dw : chest1 et (c*1*kh*kw))
  (sbias_dw : chest1 et c)
  (sw_pw : chest1 et (cout*c*1*1))
  (sbias_pw : chest1 et cout)
  (smid : chest1 et (b*c*h_out*w_out))
  : Lemma
    (requires (forall (t':nat{t' < b*c*h_out*w_out}).
                 acc1 smid t' ==
                 dwconv2d_out_at b c h_in w_in kh kw stride pad
                                 h_out w_out sx sw_dw sbias_dw t'))
    (ensures (forall (tid:nat{tid < b*cout*h_out*w_out}).
                conv2d_out_at b c h_out w_out cout 1 1 1 0 h_out w_out
                              smid sw_pw sbias_pw tid ==
                separable_out_at b c h_in w_in kh kw stride pad
                                 cout h_out w_out
                                 sx sw_dw sbias_dw sw_pw sbias_pw tid))
  = mid_eq_lemma b c h_in w_in kh kw stride pad h_out w_out
      sx sw_dw sbias_dw smid;
    let xt  = lseq_to_t4 b c h_in w_in sx in
    let wt  = lseq_to_t4 c 1 kh kw sw_dw in
    let midt = lseq_to_t4 b c h_out w_out smid in
    let mat : chest2 et cout c = lseq_to_mat cout c sw_pw in
    let dwt = dwconv2d kh kw stride stride pad pad 1 1 h_out w_out
                xt wt (chest1_to_seq sbias_dw) in
    (* mid_eq_lemma's ensures gives this; bind it explicitly. *)
    assert (midt == dwt);
    introduce forall (tid:nat{tid < b*cout*h_out*w_out}).
      conv2d_out_at b c h_out w_out cout 1 1 1 0 h_out w_out
                    smid sw_pw sbias_pw tid ==
      separable_out_at b c h_in w_in kh kw stride pad
                       cout h_out w_out
                       sx sw_dw sbias_dw sw_pw sbias_pw tid
    with (
      let bi : natlt b = tid / (cout*h_out*w_out) in
      let r1 = tid % (cout*h_out*w_out) in
      let oc : natlt cout = r1 / (h_out*w_out) in
      let r2 = r1 % (h_out*w_out) in
      let oh : natlt h_out = r2 / w_out in
      let ow : natlt w_out = r2 % w_out in
      (* LHS: conv2d_out_at unfolds to a 1x1 general conv on midt.
         We inline [lseq_to_t4 cout c 1 1 sw_pw] (rather than a let-bound
         [pw4]) so the [mk2] closure produced by [pw_eq_lemma] is
         syntactically the unfolding of [mat = lseq_to_mat cout c sw_pw]
         (avoids needing function extensionality). *)
      assert (conv2d_out_at b c h_out w_out cout 1 1 1 0 h_out w_out
                            smid sw_pw sbias_pw tid ==
              conv2d_single 1 1 1 0 midt (lseq_to_t4 cout c 1 1 sw_pw)
                            (chest1_to_seq sbias_pw) bi oc oh ow);
      (* 1x1 general conv == pointwise conv. *)
      pw_eq_lemma b c h_out w_out cout midt (lseq_to_t4 cout c 1 1 sw_pw)
                  (chest1_to_seq sbias_pw) bi oc oh ow;
      assert (conv2d_single 1 1 1 0 midt (lseq_to_t4 cout c 1 1 sw_pw)
                            (chest1_to_seq sbias_pw) bi oc oh ow ==
              pwconv2d_single midt mat (chest1_to_seq sbias_pw) bi oc oh ow);
      (* midt == dwt, so the pointwise conv reads the depthwise output. *)
      assert (pwconv2d_single midt mat (chest1_to_seq sbias_pw) bi oc oh ow ==
              pwconv2d_single dwt mat (chest1_to_seq sbias_pw) bi oc oh ow);
      (* RHS: the whole separable spec chases to the same pointwise cell. *)
      lemma_separable_conv2d_index kh kw stride stride pad pad 1 1
        h_out w_out xt wt (chest1_to_seq sbias_dw) mat (chest1_to_seq sbias_pw) bi oc oh ow;
      assert (separable_out_at b c h_in w_in kh kw stride pad
                               cout h_out w_out
                               sx sw_dw sbias_dw sw_pw sbias_pw tid ==
              tacc (separable_conv2d kh kw stride stride pad pad 1 1
                      h_out w_out xt wt (chest1_to_seq sbias_dw) mat (chest1_to_seq sbias_pw))
                   bi oc oh ow)
    )

(* ================================================================== *)
(* Self-allocating composed entry.                                    *)
(* ================================================================== *)

inline_for_extraction noextract
fn separable_alloc
  (b c h_in w_in kh kw : szp)
  (stride : szp)
  (pad : sz)
  (cout : szp)
  (h_out : szp)
  (w_out : szp { separable_size_req b c h_in w_in kh kw stride cout h_out w_out })
  (gx : array1 f32 (l1_forward (b * c * h_in * w_in))
        { is_global gx })
  (gw_dw : array1 f32 (l1_forward (c * 1 * kh * kw))
        { is_global gw_dw })
  (gbias_dw : array1 f32 (l1_forward c)
        { is_global gbias_dw })
  (gw_pw : array1 f32 (l1_forward (cout * c * 1 * 1))
        { is_global gw_pw })
  (gbias_pw : array1 f32 (l1_forward cout)
        { is_global gbias_pw })
  (#fx : perm) (#fwd : perm) (#fbd : perm) (#fwp : perm) (#fbp : perm)
  (#sx : chest1 f32 (b * c * h_in * w_in))
  (#sw_dw : chest1 f32 (c * 1 * kh * kw))
  (#sbias_dw : chest1 f32 c)
  (#sw_pw : chest1 f32 (cout * c * 1 * 1))
  (#sbias_pw : chest1 f32 cout)
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw_dw |-> Frac fwd sw_dw) **
    on gpu_loc (gbias_dw |-> Frac fbd sbias_dw) **
    on gpu_loc (gw_pw |-> Frac fwp sw_pw) **
    on gpu_loc (gbias_pw |-> Frac fbp sbias_pw)
  returns gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
  ensures
    (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
               acc1 sy tid ==
               separable_out_at b c h_in w_in kh kw stride pad
                                cout h_out w_out
                                sx sw_dw sbias_dw sw_pw sbias_pw tid))
{
  (* allocate the depthwise-output scratch buffer *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v c) 1 (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v c * SZ.v h_out) 1 w_out;
  let len_mid : szp = SZ.(b *^ c *^ h_out *^ w_out);
  let gmid = alloc0 #f32 len_mid (l1_forward len_mid);
  (* depthwise stage: gmid[tid] = dwconv2d_out_at ... *)
  DW.dwconv2d_f32 b c h_in w_in kh kw stride pad h_out w_out
                  gx gw_dw gbias_dw gmid;
  with smid. assert (on gpu_loc (gmid |-> smid));
  (* allocate the final output buffer *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout) 1 (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v h_out) 1 w_out;
  let len_y : szp = SZ.(b *^ cout *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  (* pointwise (1x1) stage on the depthwise output *)
  CG.conv2d_general_f32 b c h_out w_out cout 1sz 1sz 1sz 0sz h_out w_out
                        gmid gw_pw gbias_pw gy;
  (* tie the chained posts to the whole separable spec *)
  separable_compose_lemma b c h_in w_in
    kh kw stride pad
    cout h_out w_out
    sx sw_dw sbias_dw sw_pw sbias_pw smid;
  (* free the scratch buffer *)
  free gmid;
  gy
}

let separable_alloc_f32 =
  fun b c h_in w_in kh kw stride pad cout h_out w_out
      gx gw_dw gbias_dw gw_pw gbias_pw
      #fx #fwd #fbd #fwp #fbp #sx #sw_dw #sbias_dw #sw_pw #sbias_pw ->
    separable_alloc b c h_in w_in kh kw stride pad cout h_out w_out
                    gx gw_dw gbias_dw gw_pw gbias_pw
                    #fx #fwd #fbd #fwp #fbp
                    #sx #sw_dw #sbias_dw #sw_pw #sbias_pw

inline_for_extraction let () = ()
