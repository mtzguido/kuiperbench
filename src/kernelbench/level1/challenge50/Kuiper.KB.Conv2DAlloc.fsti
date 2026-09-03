module Kuiper.KB.Conv2DAlloc

(* Self-allocating KernelBench L1 #50/#55/#56/#62 Conv2D surface.
   The ABI-facing [conv2d_raw_alloc_bias_f32] and
   [conv2d_raw_alloc_zero_f32] entries take raw dimensions, derive both output
   dimensions, create zero bias when needed, allocate the output, run the
   verified kernel, and return the dimensions with the owned buffer.  The C++
   bridge therefore performs no convolution geometry, size validation, or
   GPU staging.  Invalid raw dimensions abort at checked Pulse guards.  The
   lower-level output-size and explicit-output-dimension entries remain
   internal building blocks. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Kernel.Conv2D.Naive
open Kuiper.KB.Conv2DGeneral
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let conv2d_out_len (n : nat) (k stride : pos) (pad : nat) : nat
  = let padded = n + 2 * pad in
    if k > padded then 0 else (padded - k) / stride + 1

inline_for_extraction noextract
unfold
let conv2d_raw_size_req
  (b cin h_in w_in cout : nat) (kh kw stride : pos) (pad : nat)
  : prop
  = let h_out = conv2d_out_len h_in kh stride pad in
    let w_out = conv2d_out_len w_in kw stride pad in
    SZ.fits (h_in + 2 * pad) /\ kh <= h_in + 2 * pad /\
    SZ.fits (w_in + 2 * pad) /\ kw <= w_in + 2 * pad /\
    conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out
(* (a) Verified, extractable conv output-size formula used by the raw entry
   (see .fst). Conv2d here
   has no dilation, so the dilated kernel span equals [k] and the output
   dimension is [(n + 2*pad - k) / stride + 1].  The [requires]
   [k <= n + 2*pad] prevents size_t subtraction underflow. *)
val conv2d_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)

(* Upper bound on the conv output dimension: [out <= n + 2*pad] (see .fst). *)
val conv2d_out_dim_ub (n k stride pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ k <= n + 2 * pad)
          (ensures (n + 2 * pad - k) / stride + 1 <= n + 2 * pad)

(* (b) Self-allocating entry-point type.  Takes the raw conv dims plus
   [h_out]/[w_out] (supplied by the verified [conv2d_out_dim]).  Allocates the
   [b*cout*h_out*w_out] GPU output buffer, runs the verified kernel, and
   returns the buffer directly — ownership passes to the caller (the bridge
   wraps it in a torch tensor with a cudaFree deleter).  The post is the SAME
   per-thread [conv2d_out_at] functional spec the underlying kernel
   guarantees. *)
fn conv2d_general_alloc_f32
  (b cin h_in w_in cout kh kw stride : szp)
(pad : sz)
(h_out : szp)
(w_out : szp { conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out })
(gx : array1 f32 (l1_forward (b * cin * h_in * w_in))
     { is_global gx })
(gw : array1 f32 (l1_forward (cout * cin * kh * kw))
     { is_global gw })
(gbias : array1 f32 (l1_forward cout)
     { is_global gbias })
(#fx #fw #fb : perm)
(#sx : chest1 f32 (b * cin * h_in * w_in))
(#sw : chest1 f32 (cout * cin * kh * kw))
(#sbias : chest1 f32 cout)
preserves
 cpu **
 on gpu_loc (gx |-> Frac fx sx) **
 on gpu_loc (gw |-> Frac fw sw) **
 on gpu_loc (gbias |-> Frac fb sbias)
returns gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
ensures
 (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
    on gpu_loc (gy |-> sy) **
    pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
            acc1 sy tid ==
            conv2d_out_at b cin h_in w_in cout kh kw stride pad
                          h_out w_out sx sw sbias tid))

fn conv2d_raw_alloc_bias_f32
  (b cin h_in w_in cout kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kh * kw)) { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves
    cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns r :
    (ho : szp { SZ.v ho == conv2d_out_len h_in kh stride pad } &
     (wo : szp { SZ.v wo == conv2d_out_len w_in kw stride pad } &
      array1 f32 (l1_forward (b * cout * ho * wo))))
  ensures
    exists* (sy : chest1 f32
      (b * cout * (dfst r) * (dfst (dsnd r)))).
      on gpu_loc ((dsnd (dsnd r)) |-> sy) **
      pure (forall (tid : nat{
        tid < b * cout * (dfst r) * (dfst (dsnd r))}).
        acc1 sy tid == conv2d_out_at b cin h_in w_in cout kh kw stride pad
          (dfst r) (dfst (dsnd r)) sx sw sbias tid)

fn conv2d_raw_alloc_zero_f32
  (b cin h_in w_in cout kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kh * kw)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kh * kw))
  norewrite
  preserves
    cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns r :
    (ho : szp { SZ.v ho == conv2d_out_len h_in kh stride pad } &
     (wo : szp { SZ.v wo == conv2d_out_len w_in kw stride pad } &
      array1 f32 (l1_forward (b * cout * ho * wo))))
  ensures
    exists* (sy : chest1 f32
      (b * cout * (dfst r) * (dfst (dsnd r)))).
      on gpu_loc ((dsnd (dsnd r)) |-> sy) **
      pure (forall (tid : nat{
        tid < b * cout * (dfst r) * (dfst (dsnd r))}).
        acc1 sy tid == conv2d_out_at b cin h_in w_in cout kh kw stride pad
          (dfst r) (dfst (dsnd r)) sx sw
          (mk1 (fun _ -> (zero #f32))) tid)
