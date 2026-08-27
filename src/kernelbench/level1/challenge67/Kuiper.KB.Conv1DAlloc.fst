module Kuiper.KB.Conv1DAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv1D
open Kuiper.Kernel.Conv1D.Naive
open Kuiper.KB.Conv1DGeneral
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* (a) Verified, extractable conv1d output-size formula, provably equal to the
   pure spec [(n + 2*pad - eff_k) / stride + 1] with the dilated kernel span
   [eff_k = (k-1)*dilation + 1].  The C bridge calls this instead of
   re-implementing the division in unverified C.  Mirrors [conv2d_out_dim]
   from challenge50, with the added dilation factor.  [eff_k <= padded]
   (a precondition, discharged C-side by the "padded input >= effective
   kernel" TORCH_CHECK) keeps the size_t subtraction from underflowing;
   [fits (n + 2*pad)] keeps it in u32.  The trailing [+1] fits because
   [eff_k >= 1] gives [(padded - eff_k)/stride <= padded - 1], so the result
   is [<= padded]. *)
let conv1d_out_dim (n k stride dilation : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                (SZ.v k - 1) * SZ.v dilation + 1 <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r ==
         (SZ.v n + 2 * SZ.v pad - ((SZ.v k - 1) * SZ.v dilation + 1))
           / SZ.v stride + 1)
  =
  let padded : sz = SZ.(n +^ (2sz *^ pad)) in
  let eff_k : sz = SZ.((k -^ 1sz) *^ dilation +^ 1sz) in
  SZ.(((padded -^ eff_k) /^ stride) +^ 1sz)

(* Upper bound on the conv1d output dimension: [out <= n + 2*pad].  Since the
   dilated kernel span [eff_k >= 1] and stride [stride >= 1], we have
   [(padded - eff_k)/stride <= padded - eff_k <= padded - 1 < padded].
   Mirror of [conv2d_out_dim_ub]. *)
let conv1d_out_dim_ub (n k stride dilation pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ dilation >= 1 /\
                    (k - 1) * dilation + 1 <= n + 2 * pad)
          (ensures
             (n + 2 * pad - ((k - 1) * dilation + 1)) / stride + 1
               <= n + 2 * pad)
  = let padded = n + 2 * pad in
    let eff_k = (k - 1) * dilation + 1 in
    ML.lemma_div_mod (padded - eff_k) stride;
    assert ((padded - eff_k) / stride <= padded - eff_k)

(* (b) Self-allocating entry point.  Allocates the [b*cout*l_out] output buffer
   on the GPU via [alloc0] (extracts to cudaMalloc), runs the verified
   [conv1d_general_f32], and RETURNS the freshly-allocated buffer directly
   (binding it to a let first would sever the separation-logic resource link).
   The post forwards the full per-thread [conv1d_out_at] functional spec. *)
inline_for_extraction noextract
fn conv1d_general_alloc
  (b cin l_in cout kk : szp)
  (stride : szp)
  (pad : sz)
  (dilation : szp)
  (l_out : szp { conv1d_size_req b cin l_in cout kk stride dilation l_out })
  (gx : array1 f32 (l1_forward (b * cin * l_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kk))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 f32 (b * cin * l_in))
  (#sw : chest1 f32 (cout * cin * kk))
  (#sbias : chest1 f32 cout)
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * l_out))
  ensures
    (exists* (sy : chest1 f32 (b * cout * l_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * l_out}).
               acc1 sy tid ==
               conv1d_out_at b cin l_in cout kk stride pad dilation
                             l_out sx sw sbias tid))
{
  (* The partial product [b*cout] is bounded by the full product
     [b*cout*l_out] (every factor is [>= 1]), which fits per
     [conv1d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout) 1 l_out;
  let len_y : szp = SZ.(b *^ cout *^ l_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  conv1d_general_f32 b cin l_in cout kk stride pad dilation l_out
                     gx gw gbias gy;
  gy
}

let conv1d_general_alloc_f32 =
  fun b cin l_in cout kk stride pad dilation l_out
      gx gw gbias #fx #fw #fb #sx #sw #sbias ->
    conv1d_general_alloc b cin l_in cout kk stride pad dilation l_out
                         gx gw gbias #fx #fw #fb #sx #sw #sbias
