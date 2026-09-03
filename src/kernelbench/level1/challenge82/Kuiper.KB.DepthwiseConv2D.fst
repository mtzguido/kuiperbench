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
module Map = Kuiper.Kernel.Map
module Chest = Kuiper.Chest
module CS = Kuiper.KB.CheckedSize

inline_for_extraction noextract
let const_zero_f32 (_ : f32) : f32 = zero

let map_const_zero (#n : nat) (s : chest1 f32 n)
  : Lemma (chest_map const_zero_f32 s == mk1 (fun _ -> zero))
  = Chest.lemma_equal_intro
      (chest_map const_zero_f32 s) (mk1 (fun _ -> zero));
    Chest.ext (chest_map const_zero_f32 s) (mk1 (fun _ -> zero))

(* (a) Verified, extractable depthwise-conv output-size formula, provably
   equal to the pure spec [(n + 2*pad - k) / stride + 1].  Mirrors
   [Kuiper.KB.Conv2DAlloc.conv2d_out_dim] (depthwise dilation = 1, so the
   dilated span is just [k]).  The raw entry's checked Pulse guards establish
   [k <= padded], preventing subtraction underflow, and
   [fits (n + 2*pad)], keeping the value in u32. *)
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
  (#sx : chest1 et (b * c * h_in * w_in))
  (#sw : chest1 et (c * 1 * kh * kw))
  (#sbias : chest1 et c)
  (#sy0 : chest1 et (b * c * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
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

let dwconv2d_f32 = dwconv2d_impl #f32

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
  (#sx : chest1 f32 (b * c * h_in * w_in))
  (#sw : chest1 f32 (c * 1 * kh * kw))
  (#sbias : chest1 f32 c)
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * c * h_out * w_out))
  ensures
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
  ML.lemma_mult_le_left (SZ.v b * SZ.v c * SZ.v h_out) 1 w_out;
  let len_y : szp = SZ.(b *^ c *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  dwconv2d_impl #f32 b c h_in w_in kh kw stride pad h_out w_out
                gx gw gbias gy;
  gy
}

let dwconv2d_alloc_f32 =
  fun b c h_in w_in kh kw stride pad h_out w_out
      gx gw gbias #fx #fw #fb #sx #sw #sbias ->
    dwconv2d_alloc b c h_in w_in kh kw stride pad h_out w_out
                   gx gw gbias #fx #fw #fb #sx #sw #sbias

inline_for_extraction noextract
fn guard_dwconv2d_raw_size
  (b c h_in w_in kh kw stride : szp)
  (pad : sz)
  norewrite
  requires emp
  ensures pure (dwconv2d_raw_size_req b c h_in w_in kh kw stride pad)
{
  let two_pad = CS.mul 2sz pad;
  let h_pad = CS.addp h_in two_pad;
  let w_pad = CS.addp w_in two_pad;
  dguard (kh <=^ h_pad);
  dguard (kw <=^ w_pad);
  let h0 = dwconv2d_out_dim h_in kh stride pad;
  let w0 = dwconv2d_out_dim w_in kw stride pad;
  let h_out : szp = h0;
  let w_out : szp = w0;
  assert pure (SZ.v h_out == dwconv2d_out_len h_in kh stride pad);
  assert pure (SZ.v w_out == dwconv2d_out_len w_in kw stride pad);

  let _xlen = CS.mulp4 b c h_in w_in;
  let _klen = CS.mulp3 c kh kw;
  let _kernel = CS.mulp kh kw;
  let _hw = CS.mulp h_out w_out;
  let _chw = CS.mulp3 c h_out w_out;
  let ylen = CS.mulp4 b c h_out w_out;
  let hs = CS.mulp h_out stride;
  let _hbound = CS.addp hs kh;
  let ws = CS.mulp w_out stride;
  let _wbound = CS.addp ws kw;
  let launch_bound : szp = max_blocks *^ max_threads;
  dguard (ylen <=^ launch_bound);
  assert pure (SZ.v ylen <= SZ.v launch_bound);
}

fn dwconv2d_raw_alloc_bias_f32
  (b c h_in w_in kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * c * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (c * 1 * kh * kw)) { is_global gw })
  (gbias : array1 f32 (l1_forward c) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * c * h_in * w_in))
  (#sw : chest1 f32 (c * 1 * kh * kw))
  (#sbias : chest1 f32 c)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) ** on gpu_loc (gbias |-> Frac fb sbias)
  returns r :
    (ho : szp { SZ.v ho == dwconv2d_out_len h_in kh stride pad } &
     (wo : szp { SZ.v wo == dwconv2d_out_len w_in kw stride pad } &
      array1 f32 (l1_forward (b * c * ho * wo))))
  ensures exists* (sy : chest1 f32 (b * c * (dfst r) * (dfst (dsnd r)))).
    on gpu_loc ((dsnd (dsnd r)) |-> sy) **
    pure (forall (tid : nat{tid < b * c * (dfst r) * (dfst (dsnd r))}).
      acc1 sy tid == dwconv2d_out_at b c h_in w_in kh kw stride pad
        (dfst r) (dfst (dsnd r)) sx sw sbias tid)
{
  guard_dwconv2d_raw_size b c h_in w_in kh kw stride pad;
  let h0 = dwconv2d_out_dim h_in kh stride pad;
  let w0 = dwconv2d_out_dim w_in kw stride pad;
  assert pure (SZ.v h0 == dwconv2d_out_len h_in kh stride pad);
  assert pure (SZ.v w0 == dwconv2d_out_len w_in kw stride pad);
  let h_out : szp = h0;
  let w_out : szp = w0;
  let gy = dwconv2d_alloc_f32 b c h_in w_in kh kw stride pad
    h_out w_out gx gw gbias;
  (| h_out, (| w_out, gy |) |)
}

fn dwconv2d_raw_alloc_zero_f32
  (b c h_in w_in kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * c * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (c * 1 * kh * kw)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * c * h_in * w_in))
  (#sw : chest1 f32 (c * 1 * kh * kw))
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns r :
    (ho : szp { SZ.v ho == dwconv2d_out_len h_in kh stride pad } &
     (wo : szp { SZ.v wo == dwconv2d_out_len w_in kw stride pad } &
      array1 f32 (l1_forward (b * c * ho * wo))))
  ensures exists* (sy : chest1 f32 (b * c * (dfst r) * (dfst (dsnd r)))).
    on gpu_loc ((dsnd (dsnd r)) |-> sy) **
    pure (forall (tid : nat{tid < b * c * (dfst r) * (dfst (dsnd r))}).
      acc1 sy tid == dwconv2d_out_at b c h_in w_in kh kw stride pad
        (dfst r) (dfst (dsnd r)) sx sw (mk1 (fun _ -> (zero #f32))) tid)
{
  guard_dwconv2d_raw_size b c h_in w_in kh kw stride pad;
  let gbias = alloc0 #f32 c (l1_forward c);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 c gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> mk1 (fun _ -> (zero #f32))));
  let r = dwconv2d_raw_alloc_bias_f32 b c h_in w_in kh kw stride pad
    gx gw gbias;
  free gbias;
  r
}

inline_for_extraction let () = ()
