module Kuiper.KB.ConvT2DGeneral

(* Generic ConvTranspose2D forward surface used by KernelBench L1 #57, #65,
   #69, #71, #78, and #81.  The ABI-facing raw bias/zero entries accept the
   original dimensions and all supported asymmetric parameters, derive
   [h_out]/[w_out] including output padding, create zero bias when needed,
   allocate the result, and run the verified kernel.  Checked Pulse guards
   validate the output-padding rule and all raw size arithmetic before any
   allocation or launch. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Spec.ConvTranspose2D
open Kuiper.Kernel.ConvT2D.Naive
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let convt2d_out_len
  (n : nat) (s d k : pos) (p opad : nat) : nat
  = convT_out_len_1d n s p d k opad

noeq type convt2d_raw_result
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw oph opw : sz) (dh dw : szp) = {
  h_out : ho:szp { SZ.v ho == convt2d_out_len h_in sh dh kh ph oph };
  w_out : wo:szp { SZ.v wo == convt2d_out_len w_in sw dw kw pw opw };
  output : array1 f32 (l1_forward (b * cout * h_out * w_out));
}

inline_for_extraction noextract
unfold
let convt2d_raw_size_req
  (b cin h_in w_in cout : nat) (kh kw sh sw : pos)
  (ph pw oph opw : nat) (dh dw : pos)
  : prop
  = let h_out = convt2d_out_len h_in sh dh kh ph oph in
    let w_out = convt2d_out_len w_in sw dw kw pw opw in
    (oph < sh \/ oph < dh) /\ (opw < sw \/ opw < dw) /\
    SZ.fits ((h_in - 1) * sh + dh * (kh - 1) + oph + 1) /\
    2 * ph < (h_in - 1) * sh + dh * (kh - 1) + oph + 1 /\
    SZ.fits ((w_in - 1) * sw + dw * (kw - 1) + opw + 1) /\
    2 * pw < (w_in - 1) * sw + dw * (kw - 1) + opw + 1 /\
    convT2d_size_req b cin h_in w_in cout kh kw sh sw ph pw dh dw
      h_out w_out

(* (a) Verified, extractable ConvTranspose output-size formula used by the
   raw entries (see .fst):
   [(n-1)*s - 2*p + d*(k-1) + opad + 1].  The [requires]
   [2*p <= pos] rules out size_t underflow and is established by the raw
   entry's checked Pulse guards. *)
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

fn convt2d_general_f32
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw : sz)
  (dh dw h_out : szp)
  (w_out : szp { convT2d_size_req b cin h_in w_in cout kh kw
                                  sh sw ph pw dh dw h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
        { is_global gy })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kh * kw))
  (#sbias : chest1 f32 cout)
  (#sy0 : chest1 f32 (b * cout * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
              acc1 sy tid ==
              convT2d_out_at b cin h_in w_in cout kh kw
                             sh sw ph pw dh dw
                             h_out w_out sx sw_l sbias tid))


(* (b) Self-allocating entry-point type.  Takes the raw convT dims plus
   [h_out]/[w_out] (supplied by the verified [convt_out_dim]).  Allocates the
   [b*cout*h_out*w_out] GPU output buffer, runs the verified kernel, and
   returns the buffer directly — ownership passes to the caller (the bridge
   wraps it in a torch tensor with a cudaFree deleter).  The post is the SAME
   per-thread [convT2d_out_at] functional spec the underlying kernel
   guarantees. *)
fn convt2d_general_alloc_f32
  (b cin h_in w_in cout kh kw sh sw : szp)
(ph pw : sz)
(dh dw h_out : szp)
(w_out : szp { convT2d_size_req b cin h_in w_in cout kh kw
                               sh sw ph pw dh dw h_out w_out })
(gx : array1 f32 (l1_forward (b * cin * h_in * w_in))
     { is_global gx })
(gw : array1 f32 (l1_forward (cin * cout * kh * kw))
     { is_global gw })
(gbias : array1 f32 (l1_forward cout)
     { is_global gbias })
(#fx #fw #fb : perm)
(#sx : chest1 f32 (b * cin * h_in * w_in))
(#sw_l : chest1 f32 (cin * cout * kh * kw))
(#sbias : chest1 f32 cout)
preserves
 cpu **
 on gpu_loc (gx |-> Frac fx sx) **
 on gpu_loc (gw |-> Frac fw sw_l) **
 on gpu_loc (gbias |-> Frac fb sbias)
returns gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
ensures
 (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
    on gpu_loc (gy |-> sy) **
    pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
            acc1 sy tid ==
            convT2d_out_at b cin h_in w_in cout kh kw
                           sh sw ph pw dh dw
                           h_out w_out sx sw_l sbias tid))

inline_for_extraction noextract
fn guard_convt2d_raw_size
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw oph opw : sz)
  (dh dw : szp)
  norewrite
  requires emp
  ensures pure (convt2d_raw_size_req b cin h_in w_in cout kh kw sh sw
    ph pw oph opw dh dw)

fn convt2d_raw_alloc_bias_f32
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw oph opw : sz) (dh dw : szp)
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kh * kw)) { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) ** on gpu_loc (gbias |-> Frac fb sbias)
  returns r : convt2d_raw_result b cin h_in w_in cout kh kw sh sw
    ph pw oph opw dh dw
  ensures exists* (sy : chest1 f32
    (b * cout * r.h_out * r.w_out)).
    on gpu_loc (r.output |-> sy) **
    pure (forall (tid : nat{tid < b * cout * r.h_out * r.w_out}).
      acc1 sy tid == convT2d_out_at b cin h_in w_in cout kh kw sh sw ph pw
        dh dw r.h_out r.w_out sx sw_l sbias tid)

fn convt2d_raw_alloc_zero_f32
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw oph opw : sz) (dh dw : szp)
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kh * kw)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kh * kw))
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l)
  returns r : convt2d_raw_result b cin h_in w_in cout kh kw sh sw
    ph pw oph opw dh dw
  ensures exists* (sy : chest1 f32
    (b * cout * r.h_out * r.w_out)).
    on gpu_loc (r.output |-> sy) **
    pure (forall (tid : nat{tid < b * cout * r.h_out * r.w_out}).
      acc1 sy tid == convT2d_out_at b cin h_in w_in cout kh kw sh sw ph pw
        dh dw r.h_out r.w_out sx sw_l
        (mk1 (fun _ -> (zero #f32))) tid)


inline_for_extraction let () = ()
