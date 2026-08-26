module Kuiper.KB.DepthwiseConv2D

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Spec.DepthwiseConv2D
open Kuiper.Kernel.Conv2D.Depthwise
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* (a) Verified, extractable depthwise-conv output-size formula, provably
   equal to the pure spec [(n + 2*pad - k) / stride + 1].  Mirrors
   [Kuiper.KB.Conv2DAlloc.conv2d_out_dim] (depthwise dilation = 1, so the
   dilated span is just [k]).  [k <= padded] (precondition, discharged C-side
   by the "padded input >= kernel" TORCH_CHECK) keeps the size_t subtraction
   from underflowing; [fits (n + 2*pad)] keeps it in u32. *)
let dwconv2d_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)
  =
  let padded : sz = SZ.(n +^ (2sz *^ pad)) in
  SZ.(((padded -^ k) /^ stride) +^ 1sz)

(* Upper bound on the depthwise-conv output dimension: [out <= n + 2*pad]. *)
let dwconv2d_out_dim_ub (n k stride pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ k <= n + 2 * pad)
          (ensures (n + 2 * pad - k) / stride + 1 <= n + 2 * pad)
  = let padded = n + 2 * pad in
    ML.lemma_div_mod (padded - k) stride;
    assert ((padded - k) / stride <= padded - k)

inline_for_extraction noextract
fn dwconv2d_impl
  (#et : Type0) {| scalar et |}
  (b c h_in w_in kh kw : szp)
  (stride : szp)
  (pad : sz)
  (h_out : szp)
  (w_out : szp { dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out })
  (gx : array1 et (l1_forward (b * c * h_in * w_in))
        { is_global gx })
  (gw : array1 et (l1_forward (c * 1 * kh * kw))
        { is_global gw })
  (gbias : array1 et (l1_forward c)
        { is_global gbias })
  (gy : array1 et (l1_forward (b * c * h_out * w_out))
        { is_global gy })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : erased (chest1 et (b * c * h_in * w_in)))
  (#sw : erased (chest1 et (c * 1 * kh * kw)))
  (#sbias : erased (chest1 et c))
  (#sy0 : erased (chest1 et (b * c * h_out * w_out)))
  norewrite
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b * c * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * c * h_out * w_out}).
               acc1 sy tid ==
               dwconv2d_out_at b c h_in w_in kh kw stride pad
                               h_out w_out sx sw sbias tid))
{
  dwconv2d_naive_gpu #et b c h_in w_in kh kw stride pad h_out w_out
                     gx gw gbias gy;
  ()
}

let dwconv2d_f32 : dwconv2d_ty f32 = dwconv2d_impl #f32

(* (b) Self-allocating entry point.  Allocates the [b*c*h_out*w_out] output
   buffer on the GPU via [alloc0] (extracts to cudaMalloc), runs the
   verified [dwconv2d_naive_gpu], and RETURNS the freshly-allocated buffer
   directly (binding it to a let first would sever the separation-logic
   resource link).  The post forwards the full per-thread [dwconv2d_out_at]
   functional spec.  Mirror of [Kuiper.KB.Conv2DAlloc.conv2d_general_alloc]. *)
inline_for_extraction noextract
fn dwconv2d_alloc
  (b c h_in w_in kh kw : szp)
  (stride : szp)
  (pad : sz)
  (h_out : szp)
  (w_out : szp { dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out })
  (gx : array1 f32 (l1_forward (b * c * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (c * 1 * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward c)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : erased (chest1 f32 (b * c * h_in * w_in)))
  (#sw : erased (chest1 f32 (c * 1 * kh * kw)))
  (#sbias : erased (chest1 f32 c))
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * c * h_out * w_out))
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 f32 (b * c * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * c * h_out * w_out}).
               acc1 sy tid ==
               dwconv2d_out_at b c h_in w_in kh kw stride pad
                               h_out w_out sx sw sbias tid))
{
  (* All partial products of [b*c*h_out*w_out] are bounded by the full
     product (every factor is [>= 1]), which fits per [dwconv2d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v c) 1 (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v c * SZ.v h_out) 1 (SZ.v w_out);
  let len_y : szp = SZ.(b *^ c *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  dwconv2d_impl #f32 b c h_in w_in kh kw stride pad h_out w_out
                gx gw gbias gy;
  gy
}

let dwconv2d_alloc_f32 : dwconv2d_alloc_ty =
  fun b c h_in w_in kh kw stride pad h_out w_out
      gx gw gbias #fx #fw #fb #sx #sw #sbias ->
    dwconv2d_alloc b c h_in w_in kh kw stride pad h_out w_out
                   gx gw gbias #fx #fw #fb #sx #sw #sbias

inline_for_extraction let () = ()
