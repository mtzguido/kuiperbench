module Kuiper.KB.Conv2DAlloc

(* Self-allocating bridge layer for KernelBench L1 #50/#55/#56/#62 — Conv2D
   forward (general).  Moves TWO pieces of arithmetic/allocation that used to
   live in UNVERIFIED C++ inside the verification boundary:

     1. [conv2d_out_dim]: the conv output-size division
        [(n + 2*pad - k) / stride + 1], a VERIFIED extractable SZ-level
        formula.  The C bridge calls this instead of re-implementing the
        formula.

     2. [conv2d_general_alloc_f32]: a self-allocating entry point that
        allocates the [b*cout*h_out*w_out] output buffer on the GPU via
        [alloc0] (extracts to cudaMalloc), runs the verified
        [Kuiper.KB.Conv2DGeneral.conv2d_general_f32], and returns the
        freshly-allocated buffer DIRECTLY.  Its post forwards the FULL
        functional [conv2d_out_at] spec the underlying kernel guarantees. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Kernel.Conv2D.Naive
open Kuiper.KB.Conv2DGeneral
module SZ = Kuiper.SizeT
(* (a) Verified, extractable conv output-size formula (see .fst). Conv2d here
   has no dilation, so the dilated kernel span equals [k] and the output
   dimension is [(n + 2*pad - k) / stride + 1].  The [requires]
   [k <= n + 2*pad] is exactly the C-side "padded input >= kernel" check
   (so the size_t subtraction does not underflow). *)
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
inline_for_extraction noextract
type conv2d_general_alloc_ty =
  (b : szp) ->
  (cin : szp) ->
  (h_in : szp) ->
  (w_in : szp) ->
  (cout : szp) ->
  (kh : szp) ->
  (kw : szp) ->
  (stride : szp) ->
  (pad : sz) ->
  (h_out : szp) ->
  (w_out : szp { conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out }) ->
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in))
        { is_global gx }) ->
  (gw : array1 f32 (l1_forward (cout * cin * kh * kw))
        { is_global gw }) ->
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias }) ->
  (#fx : perm) -> (#fw : perm) -> (#fb : perm) ->
  (#sx : erased (chest1 f32 (b * cin * h_in * w_in))) ->
  (#sw : erased (chest1 f32 (cout * cin * kh * kw))) ->
  (#sbias : erased (chest1 f32 cout)) ->
  stt (array1 f32 (l1_forward (b * cout * h_out * w_out)))
    (requires
       cpu **
       on gpu_loc (gx |-> Frac fx sx) **
       on gpu_loc (gw |-> Frac fw sw) **
       on gpu_loc (gbias |-> Frac fb sbias))
    (ensures fun gy ->
       cpu **
       on gpu_loc (gx |-> Frac fx sx) **
       on gpu_loc (gw |-> Frac fw sw) **
       on gpu_loc (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
          on gpu_loc (gy |-> sy) **
          pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
                  acc1 sy tid ==
                  conv2d_out_at b cin h_in w_in cout kh kw stride pad
                                h_out w_out sx sw sbias tid)))

val conv2d_general_alloc_f32 : conv2d_general_alloc_ty
