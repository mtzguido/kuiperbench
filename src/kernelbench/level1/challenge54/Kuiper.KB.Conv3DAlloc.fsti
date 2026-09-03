module Kuiper.KB.Conv3DAlloc

(* Self-allocating KernelBench L1 #54/#59/#60/#66 Conv3D surface.
   The ABI-facing [conv3d_raw_alloc_bias_f32] and
   [conv3d_raw_alloc_zero_f32] entries take raw dimensions, derive all three
   output dimensions, create zero bias when needed, allocate the output, run
   the verified kernel, and return the dimensions with the owned buffer.
   Checked Pulse guards validate all raw size arithmetic before allocation or
   launch; the C++ bridge performs no convolution geometry or GPU staging. The
   lower-level helpers remain internal building blocks. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Kernel.Conv3D.Naive
open Kuiper.KB.Conv3DGeneral
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let conv3d_out_len (n : nat) (k stride : pos) (pad : nat) : nat
  = let padded = n + 2 * pad in
    if k > padded then 0 else (padded - k) / stride + 1

inline_for_extraction noextract
unfold
let conv3d_raw_size_req
  (b cin d_in h_in w_in cout : nat) (kd kh kw stride : pos) (pad : nat)
  : prop
  = let d_out = conv3d_out_len d_in kd stride pad in
    let h_out = conv3d_out_len h_in kh stride pad in
    let w_out = conv3d_out_len w_in kw stride pad in
    SZ.fits (d_in + 2 * pad) /\ kd <= d_in + 2 * pad /\
    SZ.fits (h_in + 2 * pad) /\ kh <= h_in + 2 * pad /\
    SZ.fits (w_in + 2 * pad) /\ kw <= w_in + 2 * pad /\
    conv3d_size_req b cin d_in h_in w_in cout kd kh kw stride
      d_out h_out w_out

(* Name the dependent result and its ownership predicate.  Keeping this
   boundary named is important for verification performance: expanding the
   five-dimensional postcondition at every internal call creates thousands of
   duplicate refinement goals. *)
unfold
let conv3d_raw_result
  (b cin d_in h_in w_in cout kd kh kw stride : szp) (pad : sz) : Type0 =
  (do_ : szp { SZ.v do_ == conv3d_out_len d_in kd stride pad } &
   (ho : szp { SZ.v ho == conv3d_out_len h_in kh stride pad } &
    (wo : szp { SZ.v wo == conv3d_out_len w_in kw stride pad } &
     array1 f32 (l1_forward (b * cout * do_ * ho * wo)))))

unfold
let conv3d_raw_post
  (b cin d_in h_in w_in cout kd kh kw stride : szp) (pad : sz)
  (sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (sw : chest1 f32 (cout * cin * kd * kh * kw))
  (sbias : chest1 f32 cout)
  (r : conv3d_raw_result b cin d_in h_in w_in cout kd kh kw stride pad)
  : slprop =
  exists* (sy : chest1 f32 (b * cout * (dfst r) *
    (dfst (dsnd r)) * (dfst (dsnd (dsnd r))))).
    on gpu_loc ((dsnd (dsnd (dsnd r))) |-> sy) **
    pure (forall (tid : nat{tid < b * cout * (dfst r) *
      (dfst (dsnd r)) * (dfst (dsnd (dsnd r)))}).
      acc1 sy tid == conv3d_out_at b cin d_in h_in w_in cout kd kh kw
        stride pad (dfst r) (dfst (dsnd r)) (dfst (dsnd (dsnd r)))
        sx sw sbias tid)
(* (a) Verified, extractable conv3d output-size formula used by the raw entry
   (see .fst).  Conv3d
   here has dilation fixed to 1, so the dilated kernel span equals [k] and the
   output dimension is [(n + 2*pad - k) / stride + 1].  The [requires]
   [k <= n + 2*pad] prevents size_t subtraction underflow. *)
val conv3d_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)

(* Upper bound on the conv3d output dimension: [out <= n + 2*pad] (see .fst). *)
val conv3d_out_dim_ub (n k stride pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ k <= n + 2 * pad)
          (ensures (n + 2 * pad - k) / stride + 1 <= n + 2 * pad)

(* (b) Self-allocating entry-point type.  Takes the raw conv3d dims plus
   [d_out]/[h_out]/[w_out] (supplied by the verified [conv3d_out_dim]).
   Allocates the [b*cout*d_out*h_out*w_out] GPU output buffer, runs the
   verified kernel, and returns the buffer directly — ownership passes to the
   caller (the bridge wraps it in a torch tensor with a cudaFree deleter).
   The post is the SAME per-thread [conv3d_out_at] functional spec the
   underlying kernel guarantees. *)
fn conv3d_general_alloc_f32
  (b cin d_in h_in w_in cout kd kh kw stride : szp)
(pad : sz)
(d_out h_out : szp)
(w_out : szp { conv3d_size_req b cin d_in h_in w_in cout kd kh kw stride
                               d_out h_out w_out })
(gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
     { is_global gx })
(gw : array1 f32 (l1_forward (cout * cin * kd * kh * kw))
     { is_global gw })
(gbias : array1 f32 (l1_forward cout)
     { is_global gbias })
(#fx #fw #fb : perm)
(#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
(#sw : chest1 f32 (cout * cin * kd * kh * kw))
(#sbias : chest1 f32 cout)
norewrite
preserves
 cpu **
 on gpu_loc (gx |-> Frac fx sx) **
 on gpu_loc (gw |-> Frac fw sw) **
 on gpu_loc (gbias |-> Frac fb sbias)
returns gy : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out))
ensures
 (exists* (sy : chest1 f32 (b * cout * d_out * h_out * w_out)).
    on gpu_loc (gy |-> sy) **
    pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
            acc1 sy tid ==
            conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                          d_out h_out w_out sx sw sbias tid))

inline_for_extraction noextract
fn guard_conv3d_raw_size
  (b cin d_in h_in w_in cout kd kh kw stride : szp)
  (pad : sz)
  norewrite
  requires emp
  ensures pure (conv3d_raw_size_req b cin d_in h_in w_in cout kd kh kw
    stride pad)

fn conv3d_raw_alloc_bias_f32
  (b cin d_in h_in w_in cout kd kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
    { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kd * kh * kw))
    { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) ** on gpu_loc (gbias |-> Frac fb sbias)
  returns r : conv3d_raw_result b cin d_in h_in w_in cout kd kh kw
    stride pad
  ensures conv3d_raw_post b cin d_in h_in w_in cout kd kh kw stride pad
    sx sw sbias r

fn conv3d_raw_alloc_zero_f32
  (b cin d_in h_in w_in cout kd kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
    { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kd * kh * kw))
    { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kd * kh * kw))
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns r : conv3d_raw_result b cin d_in h_in w_in cout kd kh kw
    stride pad
  ensures conv3d_raw_post b cin d_in h_in w_in cout kd kh kw stride pad
    sx sw (mk1 (fun _ -> (zero #f32))) r
