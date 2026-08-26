module Kuiper.KB.Conv3DAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Kernel.Conv3D.Naive
open Kuiper.KB.Conv3DGeneral
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* (a) Verified, extractable conv3d output-size formula, provably equal to the
   pure spec [(n + 2*pad - k) / stride + 1].  Conv3d here has dilation fixed to
   1 (the only mode the four upstream tests exercise), so the dilated kernel
   span equals [k].  The C bridge calls this — once per spatial axis (D/H/W) —
   instead of re-implementing the division in unverified C.  Identical in shape
   to [conv2d_out_dim] from challenge50.  [k <= padded] (a precondition,
   discharged C-side by the "padded input >= kernel" TORCH_CHECK) keeps the
   size_t subtraction from underflowing; [fits (n + 2*pad)] keeps it in u32. *)
let conv3d_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)
  =
  let padded : sz = SZ.(n +^ (2sz *^ pad)) in
  SZ.(((padded -^ k) /^ stride) +^ 1sz)

(* Upper bound on the conv3d output dimension: [out <= n + 2*pad].  Since the
   kernel span [k >= 1] and stride [stride >= 1], we have
   [(padded - k)/stride <= padded - k <= padded - 1 < padded].  Mirror of
   [conv2d_out_dim_ub]. *)
let conv3d_out_dim_ub (n k stride pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ k <= n + 2 * pad)
          (ensures (n + 2 * pad - k) / stride + 1 <= n + 2 * pad)
  = let padded = n + 2 * pad in
    ML.lemma_div_mod (padded - k) stride;
    assert ((padded - k) / stride <= padded - k)

(* (b) Self-allocating entry point.  Allocates the [b*cout*d_out*h_out*w_out]
   output buffer on the GPU via [alloc0] (extracts to cudaMalloc), runs
   the verified [conv3d_general_f32], and RETURNS the freshly-allocated buffer
   directly (binding it to a let first would sever the separation-logic
   resource link).  The post forwards the full per-thread [conv3d_out_at]
   functional spec. *)
inline_for_extraction noextract
fn conv3d_general_alloc
  (b cin d_in h_in w_in cout kd kh kw : szp)
  (stride : szp)
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
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : erased (chest1 f32 (b * cin * d_in * h_in * w_in)))
  (#sw : erased (chest1 f32 (cout * cin * kd * kh * kw)))
  (#sbias : erased (chest1 f32 cout))
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
{
  (* All partial products of [b*cout*d_out*h_out*w_out] are bounded by the full
     product (every factor is [>= 1]), which fits per [conv3d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout) 1
    (SZ.v d_out * SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v d_out) 1
    (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v d_out * SZ.v h_out) 1
    (SZ.v w_out);
  let len_y : szp = SZ.(b *^ cout *^ d_out *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  conv3d_general_f32 b cin d_in h_in w_in cout kd kh kw stride pad
                     d_out h_out w_out gx gw gbias gy;
  gy
}

let conv3d_general_alloc_f32 : conv3d_general_alloc_ty =
  fun b cin d_in h_in w_in cout kd kh kw stride pad d_out h_out w_out
      gx gw gbias #fx #fw #fb #sx #sw #sbias ->
    conv3d_general_alloc b cin d_in h_in w_in cout kd kh kw stride pad
                         d_out h_out w_out gx gw gbias
                         #fx #fw #fb #sx #sw #sbias
