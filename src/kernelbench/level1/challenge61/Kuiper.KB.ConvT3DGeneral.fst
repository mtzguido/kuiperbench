module Kuiper.KB.ConvT3DGeneral

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Spec.ConvTranspose3D
open Kuiper.Kernel.ConvT3D.Naive
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* (a) Verified, extractable ConvTranspose output-size formula (per spatial
   axis), provably equal to the PyTorch formula
   [(n-1)*s - 2*p + d*(k-1) + opad + 1].  The C bridge calls this instead of
   re-implementing the arithmetic in unverified C++.  Mirrors
   [conv2d_out_dim] from challenge50 / [convt_out_dim] from challenge57. *)
let convt_out_dim (n s d k : szp) (p opad : sz)
  : Pure SZ.t
      (requires
         SZ.fits ((SZ.v n - 1) * SZ.v s + SZ.v d * (SZ.v k - 1)
                  + SZ.v opad + 1) /\
         2 * SZ.v p <= (SZ.v n - 1) * SZ.v s + SZ.v d * (SZ.v k - 1)
                       + SZ.v opad + 1)
      (ensures fun r ->
         SZ.v r == (SZ.v n - 1) * SZ.v s - 2 * SZ.v p
                   + SZ.v d * (SZ.v k - 1) + SZ.v opad + 1)
  =
  let pos : sz = SZ.((n -^ 1sz) *^ s +^ d *^ (k -^ 1sz) +^ opad +^ 1sz) in
  SZ.(pos -^ (2sz *^ p))

(* Upper bound on the ConvTranspose output dimension (see challenge57). *)
let convt_out_dim_ub (n s d k p opad : nat)
  : Lemma (requires n >= 1 /\ k >= 1)
          (ensures (n - 1) * s - 2 * p + d * (k - 1) + opad + 1
                   <= (n - 1) * s + d * (k - 1) + opad + 1)
  = ()

inline_for_extraction noextract
fn convt3d_general_impl
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (sd sh sw : szp) (pd ph pw : sz) (dd dh dw : szp)
  (d_out h_out : szp)
  (w_out : szp { convT3d_size_req b cin d_in h_in w_in cout kd kh kw
                                  sd sh sw pd ph pw dd dh dw
                                  d_out h_out w_out })
  (gx : array1 et (l1_forward (b * cin * d_in * h_in * w_in))
        { is_global gx })
  (gw : array1 et (l1_forward (cin * cout * kd * kh * kw))
        { is_global gw })
  (gbias : array1 et (l1_forward cout)
        { is_global gbias })
  (gy : array1 et (l1_forward (b * cout * d_out * h_out * w_out))
        { is_global gy })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 et (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 et (cin * cout * kd * kh * kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b * cout * d_out * h_out * w_out))
  norewrite
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b * cout * d_out * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
               acc1 sy tid ==
               convT3d_out_at b cin d_in h_in w_in cout kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias tid))
{
  convt3d_naive_gpu #et b cin d_in h_in w_in cout kd kh kw sd sh sw
                    pd ph pw dd dh dw d_out h_out w_out gx gw gbias gy;
  ()
}

let convt3d_general_f32 : convt3d_general_ty f32 = convt3d_general_impl #f32

(* (b) Self-allocating entry point.  Allocates the
   [b*cout*d_out*h_out*w_out] output buffer on the GPU via [alloc0]
   (extracts to cudaMalloc), runs the verified [convt3d_general_f32], and
   RETURNS the freshly-allocated buffer directly.  Mirrors
   [convt2d_general_alloc] from challenge57. *)
inline_for_extraction noextract
fn convt3d_general_alloc
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (sd sh sw : szp) (pd ph pw : sz) (dd dh dw : szp)
  (d_out h_out : szp)
  (w_out : szp { convT3d_size_req b cin d_in h_in w_in cout kd kh kw
                                  sd sh sw pd ph pw dd dh dw
                                  d_out h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kd * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out))
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 f32 (b * cout * d_out * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
               acc1 sy tid ==
               convT3d_out_at b cin d_in h_in w_in cout kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias tid))
{
  (* All partial products of [b*cout*d_out*h_out*w_out] are bounded by the
     full product (every factor is [>= 1]), which fits per
     [convT3d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout)
                        1 (SZ.v d_out * SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v d_out)
                        1 (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v d_out * SZ.v h_out)
                        1 w_out;
  let len_y : szp = SZ.(b *^ cout *^ d_out *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  convt3d_general_f32 b cin d_in h_in w_in cout kd kh kw sd sh sw
                      pd ph pw dd dh dw d_out h_out w_out
                      gx gw gbias gy;
  gy
}

let convt3d_general_alloc_f32 : convt3d_general_alloc_ty =
  fun b cin d_in h_in w_in cout kd kh kw sd sh sw pd ph pw dd dh dw
      d_out h_out w_out gx gw gbias #fx #fw #fb #sx #sw_l #sbias ->
    convt3d_general_alloc b cin d_in h_in w_in cout kd kh kw sd sh sw
                          pd ph pw dd dh dw d_out h_out w_out
                          gx gw gbias
                          #fx #fw #fb #sx #sw_l #sbias

inline_for_extraction let () = ()
