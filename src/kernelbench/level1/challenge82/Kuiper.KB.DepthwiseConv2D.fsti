module Kuiper.KB.DepthwiseConv2D

(* KernelBench L1 #82..#85 (depthwise 2D convolution forward) entry point.

   Wraps [Kuiper.Kernel.Conv2D.Depthwise.dwconv2d_naive_gpu] for the
   depthwise regime [nn.Conv2d(C, C, ..., groups = C)] with scalar
   stride / pad (KB tests all use stride=1, pad=0, dilation=1).

   The caller passes the depthwise weight (C, 1, kH, kW) and an optional
   bias.  The bridge supplies a zero-bias scratch buffer when bias=False
   (the verified entry point requires a bias array of length C). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Spec.DepthwiseConv2D
open Kuiper.Kernel.Conv2D.Depthwise
module SZ = Kuiper.SizeT

(* (a) Verified, extractable depthwise-conv output-size formula (see .fst).
   Depthwise conv here has dilation = 1, so the dilated kernel span equals
   [k] and the output dimension is [(n + 2*pad - k) / stride + 1].  Identical
   arithmetic to [Kuiper.KB.Conv2DAlloc.conv2d_out_dim]; replicated here so the
   depthwise bridge calls a helper extracted into its OWN translation unit
   ([Kuiper_KB_DepthwiseConv2D.cu]).  The [requires] [k <= n + 2*pad] is the
   C-side "padded input >= kernel" check (so the size_t subtraction does not
   underflow). *)
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

inline_for_extraction noextract
type dwconv2d_ty (t:Type0) {| scalar t |} =
  fn (b : szp)
     (c : szp)
     (h_in : szp)
     (w_in : szp)
     (kh : szp)
     (kw : szp)
     (stride : szp)
     (pad : sz)
     (h_out : szp)
     (w_out : szp { dwconv2d_size_req b c h_in w_in kh kw stride h_out w_out })
     (gx : array1 t (l1_forward (b * c * h_in * w_in))
           { is_global gx })
     (gw : array1 t (l1_forward (c * 1 * kh * kw))
           { is_global gw })
     (gbias : array1 t (l1_forward c)
           { is_global gbias })
     (gy : array1 t (l1_forward (b * c * h_out * w_out))
           { is_global gy })
     (#fx : perm) (#fw : perm) (#fb : perm)
     (#sx : erased (chest1 t (b * c * h_in * w_in)))
     (#sw : erased (chest1 t (c * 1 * kh * kw)))
     (#sbias : erased (chest1 t c))
     (#sy0 : erased (chest1 t (b * c * h_out * w_out)))
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
       (exists* (sy : chest1 t (b * c * h_out * w_out)).
         on gpu_loc (gy |-> sy) **
         pure (forall (tid : nat{tid < b * c * h_out * w_out}).
                 acc1 sy tid ==
                 dwconv2d_out_at b c h_in w_in kh kw stride pad
                                 h_out w_out sx sw sbias tid))

val dwconv2d_f32 : dwconv2d_ty f32

(* (b) Self-allocating entry-point type.  Takes the raw depthwise-conv dims
   plus [h_out]/[w_out] (supplied by the verified [dwconv2d_out_dim]).
   Allocates the [b*c*h_out*w_out] GPU output buffer, runs the verified
   kernel, and returns the buffer directly — ownership passes to the caller
   (the bridge wraps it in a torch tensor with a cudaFree deleter).  The post
   is the SAME per-thread [dwconv2d_out_at] functional spec the underlying
   kernel guarantees. *)
inline_for_extraction noextract
type dwconv2d_alloc_ty =
  fn
  (b : szp)
  (c : szp)
  (h_in : szp)
  (w_in : szp)
  (kh : szp)
  (kw : szp)
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

val dwconv2d_alloc_f32 : dwconv2d_alloc_ty

inline_for_extraction let () = ()
