module Kuiper.KB.Conv2DAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Kernel.Conv2D.Naive
open Kuiper.KB.Conv2DGeneral
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* (a) Verified, extractable conv output-size formula, provably equal to the
   pure spec [(n + 2*pad - k) / stride + 1].  The C bridge calls this instead
   of re-implementing the division in unverified C.  Mirrors
   [pool_out_len_1d_sz] from challenge44.  [k <= padded] (a precondition,
   discharged C-side by the "padded input >= kernel" TORCH_CHECK) keeps the
   size_t subtraction from underflowing; [fits (n + 2*pad)] keeps it in u32.
   The trailing [+1] fits because [k >= 1] gives
   [(padded - k)/stride <= padded - 1], so the result is [<= padded]. *)
let conv2d_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)
  =
  let padded : sz = SZ.(n +^ (2sz *^ pad)) in
  SZ.(((padded -^ k) /^ stride) +^ 1sz)

(* Upper bound on the conv output dimension: [out <= n + 2*pad].  Since the
   kernel span [k >= 1] and stride [stride >= 1], we have
   [(padded - k)/stride <= padded - k <= padded - 1 < padded].  Mirror of
   [pool_out_len_1d_ub]. *)
let conv2d_out_dim_ub (n k stride pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ k <= n + 2 * pad)
          (ensures (n + 2 * pad - k) / stride + 1 <= n + 2 * pad)
  = let padded = n + 2 * pad in
    ML.lemma_div_mod (padded - k) stride;
    assert ((padded - k) / stride <= padded - k)

(* (b) Self-allocating entry point.  Allocates the [b*cout*h_out*w_out] output
   buffer on the GPU via [alloc0] (extracts to cudaMalloc), runs the
   verified [conv2d_general_f32], and RETURNS the freshly-allocated buffer
   directly (binding it to a let first would sever the separation-logic
   resource link).  The post forwards the full per-thread [conv2d_out_at]
   functional spec. *)
inline_for_extraction noextract
fn conv2d_general_alloc
  (b cin h_in w_in cout kh kw : szp)
  (stride : szp)
  (pad : sz)
  (h_out : szp)
  (w_out : szp { conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
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
{
  (* All partial products of [b*cout*h_out*w_out] are bounded by the full
     product (every factor is [>= 1]), which fits per [conv2d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout) 1 (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v h_out) 1 w_out;
  let len_y : szp = SZ.(b *^ cout *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  conv2d_general_f32 b cin h_in w_in cout kh kw stride pad h_out w_out
                     gx gw gbias gy;
  gy
}

let conv2d_general_alloc_f32 =
  fun b cin h_in w_in cout kh kw stride pad h_out w_out
      gx gw gbias #fx #fw #fb #sx #sw #sbias ->
    conv2d_general_alloc b cin h_in w_in cout kh kw stride pad h_out w_out
                         gx gw gbias #fx #fw #fb #sx #sw #sbias
