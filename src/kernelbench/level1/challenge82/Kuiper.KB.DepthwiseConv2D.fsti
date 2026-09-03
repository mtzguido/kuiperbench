module Kuiper.KB.DepthwiseConv2D

(* KernelBench L1 #82..#85 (depthwise 2D convolution forward) entry point.

   Wraps [Kuiper.Kernel.Conv2D.Depthwise.dwconv2d_naive_gpu] for the
   depthwise regime [nn.Conv2d(C, C, ..., groups = C)] with scalar
   stride / pad (KB tests all use stride=1, pad=0, dilation=1).

   The raw entries take the depthwise weight (C, 1, kH, kW) and either a real
   bias or no bias.  The zero-bias entry creates its bias internally, derives
   both output dimensions, allocates the result, and runs the kernel.  It also
   validates all raw size arithmetic with checked Pulse guards; the C++ bridge
   does not stage a bias, compute convolution geometry, or validate products. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Spec.DepthwiseConv2D
open Kuiper.Kernel.Conv2D.Depthwise
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let dwconv2d_out_len (n : nat) (k stride : pos) (pad : nat) : nat
  = let padded = n + 2 * pad in
    if k > padded then 0 else (padded - k) / stride + 1

inline_for_extraction noextract
unfold
let dwconv2d_raw_size_req
  (b c h_in w_in : nat) (kh kw stride : pos) (pad : nat) : prop
  = let h_out = dwconv2d_out_len h_in kh stride pad in
    let w_out = dwconv2d_out_len w_in kw stride pad in
    SZ.fits (h_in + 2 * pad) /\ kh <= h_in + 2 * pad /\
    SZ.fits (w_in + 2 * pad) /\ kw <= w_in + 2 * pad /\
    dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out

(* (a) Verified, extractable depthwise-conv output-size formula used by the
   raw entry (see .fst).
   Depthwise conv here has dilation = 1, so the dilated kernel span equals
   [k] and the output dimension is [(n + 2*pad - k) / stride + 1].  Identical
   arithmetic to [Kuiper.KB.Conv2DAlloc.conv2d_out_dim].  The [requires]
   [k <= n + 2*pad] prevents size_t subtraction underflow. *)
val dwconv2d_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)

(* Upper bound on the depthwise-conv output dimension: [out <= n + 2*pad]. *)
val dwconv2d_out_dim_ub (n k stride pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ k <= n + 2 * pad)
          (ensures (n + 2 * pad - k) / stride + 1 <= n + 2 * pad)

fn dwconv2d_f32
  (b c h_in w_in kh kw stride : szp)
  (pad : sz)
  (h_out : szp)
  (w_out : szp { dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out })
  (gx : array1 f32 (l1_forward (b * c * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (c * 1 * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward c)
        { is_global gbias })
  (gy : array1 f32 (l1_forward (b * c * h_out * w_out))
        { is_global gy })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * c * h_in * w_in))
  (#sw : chest1 f32 (c * 1 * kh * kw))
  (#sbias : chest1 f32 c)
  (#sy0 : chest1 f32 (b * c * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 f32 (b * c * h_out * w_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b * c * h_out * w_out}).
              acc1 sy tid ==
              dwconv2d_out_at b c h_in w_in kh kw stride pad
                              h_out w_out sx sw sbias tid))


(* (b) Self-allocating entry-point type.  Takes the raw depthwise-conv dims
   plus [h_out]/[w_out] (supplied by the verified [dwconv2d_out_dim]).
   Allocates the [b*c*h_out*w_out] GPU output buffer, runs the verified
   kernel, and returns the buffer directly — ownership passes to the caller
   (the bridge wraps it in a torch tensor with a cudaFree deleter).  The post
   is the SAME per-thread [dwconv2d_out_at] functional spec the underlying
   kernel guarantees. *)
fn dwconv2d_alloc_f32
  (b c h_in w_in kh kw stride : szp)
(pad : sz)
(h_out : szp)
(w_out : szp { dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out })
(gx : array1 f32 (l1_forward (b * c * h_in * w_in))
     { is_global gx })
(gw : array1 f32 (l1_forward (c * 1 * kh * kw))
     { is_global gw })
(gbias : array1 f32 (l1_forward c)
     { is_global gbias })
(#fx #fw #fb : perm)
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


inline_for_extraction let () = ()
