module Kuiper.KB.ConvT3DGeneral

(* Generic ConvTranspose3D forward entry point used by KernelBench L1
   #58, #61, #68, #70, #72, #73, #77.  Exposes every parameter that the
   underlying [Kuiper.Kernel.ConvT3D.Naive] kernel supports: asymmetric
   inputs (d_in, h_in, w_in), asymmetric kernels (kd, kh, kw), per-axis
   stride (sd, sh, sw), pad (pd, ph, pw), dilation (dd, dh, dw).

   Groups are handled at the host (bridge) level by slicing channels and
   calling this entry point once per group: the verified primitive only
   supports the groups=1 case, which after slicing corresponds to one
   group's slab. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Spec.ConvTranspose3D
open Kuiper.Kernel.ConvT3D.Naive
module SZ = Kuiper.SizeT

(* (a) Verified, extractable ConvTranspose output-size formula (see .fst).
   The C bridge calls this instead of re-implementing
   [(n-1)*s - 2*p + d*(k-1) + opad + 1] in unverified C++. *)
val convt_out_dim (n s d k : szp) (p opad : sz)
  : Pure SZ.t
      (requires
         SZ.fits ((SZ.v n - 1) * SZ.v s + SZ.v d * (SZ.v k - 1)
                  + SZ.v opad + 1) /\
         2 * SZ.v p <= (SZ.v n - 1) * SZ.v s + SZ.v d * (SZ.v k - 1)
                       + SZ.v opad + 1)
      (ensures fun r ->
         SZ.v r == (SZ.v n - 1) * SZ.v s - 2 * SZ.v p
                   + SZ.v d * (SZ.v k - 1) + SZ.v opad + 1)

(* Upper bound on the ConvTranspose output dimension (see .fst). *)
val convt_out_dim_ub (n s d k p opad : nat)
  : Lemma (requires n >= 1 /\ k >= 1)
          (ensures (n - 1) * s - 2 * p + d * (k - 1) + opad + 1
                   <= (n - 1) * s + d * (k - 1) + opad + 1)

fn convt3d_general_f32
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw : sz)
  (dd dh dw d_out h_out : szp)
  (w_out : szp { convT3d_size_req b cin d_in h_in w_in cout kd kh kw
                                  sd sh sw pd ph pw dd dh dw
                                  d_out h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kd * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (gy : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out))
        { is_global gy })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  (#sy0 : chest1 f32 (b * cout * d_out * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 f32 (b * cout * d_out * h_out * w_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
              acc1 sy tid ==
              convT3d_out_at b cin d_in h_in w_in cout kd kh kw
                             sd sh sw pd ph pw dd dh dw
                             d_out h_out w_out sx sw_l sbias tid))


(* (b) Self-allocating entry-point type.  Takes the raw convT dims plus
   [d_out]/[h_out]/[w_out] (supplied by the verified [convt_out_dim]).
   Allocates the [b*cout*d_out*h_out*w_out] GPU output buffer, runs the
   verified kernel, and returns the buffer directly — ownership passes to the
   caller.  The post is the SAME per-thread [convT3d_out_at] functional spec
   the underlying kernel guarantees. *)
fn convt3d_general_alloc_f32
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
(pd ph pw : sz)
(dd dh dw d_out h_out : szp)
(w_out : szp { convT3d_size_req b cin d_in h_in w_in cout kd kh kw
                               sd sh sw pd ph pw dd dh dw
                               d_out h_out w_out })
(gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
     { is_global gx })
(gw : array1 f32 (l1_forward (cin * cout * kd * kh * kw))
     { is_global gw })
(gbias : array1 f32 (l1_forward cout)
     { is_global gbias })
(#fx #fw #fb : perm)
(#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
(#sw_l : chest1 f32 (cin * cout * kd * kh * kw))
(#sbias : chest1 f32 cout)
preserves
 cpu **
 on gpu_loc (gx |-> Frac fx sx) **
 on gpu_loc (gw |-> Frac fw sw_l) **
 on gpu_loc (gbias |-> Frac fb sbias)
returns gy : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out))
ensures
 (exists* (sy : chest1 f32 (b * cout * d_out * h_out * w_out)).
    on gpu_loc (gy |-> sy) **
    pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
            acc1 sy tid ==
            convT3d_out_at b cin d_in h_in w_in cout kd kh kw
                           sd sh sw pd ph pw dd dh dw
                           d_out h_out w_out sx sw_l sbias tid))


inline_for_extraction let () = ()
