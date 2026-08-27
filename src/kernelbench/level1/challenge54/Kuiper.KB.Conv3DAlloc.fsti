module Kuiper.KB.Conv3DAlloc

(* Self-allocating bridge layer for KernelBench L1 #54/#59/#60/#66 — Conv3D
   forward (general).  Moves TWO pieces of arithmetic/allocation that used to
   live in UNVERIFIED C++ inside the verification boundary:

     1. [conv3d_out_dim]: the conv3d output-size division
        [(n + 2*pad - k) / stride + 1], a VERIFIED extractable SZ-level
        formula (called once per spatial axis D/H/W).  The C bridge calls this
        instead of re-implementing the formula.

     2. [conv3d_general_alloc_f32]: a self-allocating entry point that
        allocates the [b*cout*d_out*h_out*w_out] output buffer on the GPU via
        [alloc0] (extracts to cudaMalloc), runs the verified
        [Kuiper.KB.Conv3DGeneral.conv3d_general_f32], and returns the
        freshly-allocated buffer DIRECTLY.  Its post forwards the FULL
        functional [conv3d_out_at] spec the underlying kernel guarantees. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Kernel.Conv3D.Naive
open Kuiper.KB.Conv3DGeneral
module SZ = Kuiper.SizeT
(* (a) Verified, extractable conv3d output-size formula (see .fst).  Conv3d
   here has dilation fixed to 1, so the dilated kernel span equals [k] and the
   output dimension is [(n + 2*pad - k) / stride + 1].  The [requires]
   [k <= n + 2*pad] is exactly the C-side "padded input >= kernel" check (so
   the size_t subtraction does not underflow). *)
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
inline_for_extraction noextract
type conv3d_general_alloc_ty =
  fn
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
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out))
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 f32 (b * cout * d_out * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
               acc1 sy tid ==
               conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                             d_out h_out w_out sx sw sbias tid))

val conv3d_general_alloc_f32 : conv3d_general_alloc_ty
