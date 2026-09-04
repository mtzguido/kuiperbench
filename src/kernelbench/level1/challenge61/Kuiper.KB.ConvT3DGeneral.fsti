module Kuiper.KB.ConvT3DGeneral

(* Generic groups=1 ConvTranspose3D forward surface used by KernelBench L1
   #58, #61, #68, #70, #73, and #77.  The ABI-facing raw bias/zero entries
   accept the original dimensions and asymmetric parameters, derive all
   three output dimensions including output padding, create zero bias when
   needed, allocate the result, and run the verified kernel.  Checked Pulse
   guards validate the output-padding rule and all raw size arithmetic; there
   is no host channel slicing.

   Challenge #73 is also groups=1: its nominal positional "groups" value
   binds the model's unused [output_padding] parameter.  The genuinely
   grouped fixed challenge #72 uses its separate verified specialization. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Spec.ConvTranspose3D
open Kuiper.Spec.ConvTranspose2D { convT_out_len_1d }
open Kuiper.Kernel.ConvT3D.Naive
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let convt3d_out_len
  (n : nat) (s d k : pos) (p opad : nat) : nat
  = convT_out_len_1d n s p d k opad

inline_for_extraction noextract
unfold
let convt3d_raw_size_req
  (b cin d_in h_in w_in cout : nat) (kd kh kw sd sh sw : pos)
  (pd ph pw opd oph opw : nat) (dd dh dw : pos)
  : prop
  = let d_out = convt3d_out_len d_in sd dd kd pd opd in
    let h_out = convt3d_out_len h_in sh dh kh ph oph in
    let w_out = convt3d_out_len w_in sw dw kw pw opw in
    (opd < sd \/ opd < dd) /\ (oph < sh \/ oph < dh) /\
    (opw < sw \/ opw < dw) /\
    SZ.fits ((d_in - 1) * sd + dd * (kd - 1) + opd + 1) /\
    2 * pd < (d_in - 1) * sd + dd * (kd - 1) + opd + 1 /\
    SZ.fits ((h_in - 1) * sh + dh * (kh - 1) + oph + 1) /\
    2 * ph < (h_in - 1) * sh + dh * (kh - 1) + oph + 1 /\
    SZ.fits ((w_in - 1) * sw + dw * (kw - 1) + opw + 1) /\
    2 * pw < (w_in - 1) * sw + dw * (kw - 1) + opw + 1 /\
    convT3d_size_req b cin d_in h_in w_in cout kd kh kw sd sh sw
      pd ph pw dd dh dw d_out h_out w_out

(* Preserve the three-axis dependent result as a named verification
   boundary.  Expanding this postcondition in every application causes Pulse
   to generate thousands of duplicate five-dimensional refinement goals. *)
noeq type convt3d_raw_result
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw opd oph opw : sz) (dd dh dw : szp) = {
  d_out : do_:szp { SZ.v do_ == convt3d_out_len d_in sd dd kd pd opd };
  h_out : ho:szp { SZ.v ho == convt3d_out_len h_in sh dh kh ph oph };
  w_out : wo:szp { SZ.v wo == convt3d_out_len w_in sw dw kw pw opw };
  output : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out));
}

unfold
let convt3d_raw_post
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw opd oph opw : sz) (dd dh dw : szp)
  (sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (sw_l : chest1 f32 (cin * cout * kd * kh * kw))
  (sbias : chest1 f32 cout)
  (r : convt3d_raw_result b cin d_in h_in w_in cout kd kh kw sd sh sw
    pd ph pw opd oph opw dd dh dw) : slprop =
  exists* (sy : chest1 f32 (b * cout * r.d_out * r.h_out * r.w_out)).
    on gpu_loc (r.output |-> sy) **
    pure (forall (tid : nat{tid < b * cout * r.d_out * r.h_out * r.w_out}).
      acc1 sy tid == convT3d_out_at b cin d_in h_in w_in cout kd kh kw
        sd sh sw pd ph pw dd dh dw r.d_out r.h_out r.w_out
        sx sw_l sbias tid)

(* (a) Verified, extractable ConvTranspose output-size formula used by the
   raw entries (see .fst):
   [(n-1)*s - 2*p + d*(k-1) + opad + 1]. *)
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
norewrite
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

inline_for_extraction noextract
fn guard_convt3d_raw_size
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw opd oph opw : sz)
  (dd dh dw : szp)
  norewrite
  requires emp
  ensures pure (convt3d_raw_size_req b cin d_in h_in w_in cout kd kh kw
    sd sh sw pd ph pw opd oph opw dd dh dw)

fn convt3d_raw_alloc_bias_f32
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw opd oph opw : sz) (dd dh dw : szp)
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
    { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kd * kh * kw))
    { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) ** on gpu_loc (gbias |-> Frac fb sbias)
  returns r : convt3d_raw_result b cin d_in h_in w_in cout kd kh kw
    sd sh sw pd ph pw opd oph opw dd dh dw
  ensures convt3d_raw_post b cin d_in h_in w_in cout kd kh kw sd sh sw
    pd ph pw opd oph opw dd dh dw sx sw_l sbias r

fn convt3d_raw_alloc_zero_f32
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw opd oph opw : sz) (dd dh dw : szp)
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
    { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kd * kh * kw))
    { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kd * kh * kw))
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l)
  returns r : convt3d_raw_result b cin d_in h_in w_in cout kd kh kw
    sd sh sw pd ph pw opd oph opw dd dh dw
  ensures convt3d_raw_post b cin d_in h_in w_in cout kd kh kw sd sh sw
    pd ph pw opd oph opw dd dh dw sx sw_l
    (mk1 (fun _ -> (zero #f32))) r


inline_for_extraction let () = ()
