module Kuiper.KB.ConvT2DGeneral

(* Generic ConvTranspose2D forward entry point used by KernelBench L1
   #57, #65, #69, #71 (and other ConvT2D variants).  Exposes every
   parameter that the underlying [Kuiper.Kernel.ConvT2D.Naive] kernel
   supports: asymmetric inputs (h_in, w_in), asymmetric kernels
   (kh, kw), per-axis stride (sh, sw), pad (ph, pw), dilation (dh, dw).
   The caller passes the bias array directly (and is responsible for
   choosing [h_out, w_out] consistent with PyTorch's output_padding
   formula). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Spec.ConvTranspose2D
open Kuiper.Kernel.ConvT2D.Naive
module SZ = Kuiper.SizeT

(* (a) Verified, extractable ConvTranspose output-size formula (see .fst).
   The C bridge calls this instead of re-implementing
   [(n-1)*s - 2*p + d*(k-1) + opad + 1] in unverified C++.  The [requires]
   [2*p <= pos] (the non-negative part of the formula) is exactly the C-side
   "output dim >= 1" check, ruling out size_t underflow. *)
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

inline_for_extraction noextract
type convt2d_general_ty (t:Type0) {| scalar t |} =
  fn (b cin h_in w_in cout kh kw sh sw : szp)
     (ph pw : sz)
     (dh dw h_out : szp)
     (w_out : szp { convT2d_size_req b cin h_in w_in cout kh kw
                                     sh sw ph pw dh dw h_out w_out })
     (gx : array1 t (l1_forward (b * cin * h_in * w_in))
           { is_global gx })
     (gw : array1 t (l1_forward (cin * cout * kh * kw))
           { is_global gw })
     (gbias : array1 t (l1_forward cout)
           { is_global gbias })
     (gy : array1 t (l1_forward (b * cout * h_out * w_out))
           { is_global gy })
     (#fx #fw #fb : perm)
     (#sx : chest1 t (b * cin * h_in * w_in))
     (#sw_l : chest1 t (cin * cout * kh * kw))
     (#sbias : chest1 t cout)
     (#sy0 : chest1 t (b * cout * h_out * w_out))
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
       (exists* (sy : chest1 t (b * cout * h_out * w_out)).
         on gpu_loc (gy |-> sy) **
         pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
                 acc1 sy tid ==
                 convT2d_out_at b cin h_in w_in cout kh kw
                                sh sw ph pw dh dw
                                h_out w_out sx sw_l sbias tid))

val convt2d_general_f32 : convt2d_general_ty f32

(* (b) Self-allocating entry-point type.  Takes the raw convT dims plus
   [h_out]/[w_out] (supplied by the verified [convt_out_dim]).  Allocates the
   [b*cout*h_out*w_out] GPU output buffer, runs the verified kernel, and
   returns the buffer directly — ownership passes to the caller (the bridge
   wraps it in a torch tensor with a cudaFree deleter).  The post is the SAME
   per-thread [convT2d_out_at] functional spec the underlying kernel
   guarantees. *)
inline_for_extraction noextract
type convt2d_general_alloc_ty =
  fn
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
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
               acc1 sy tid ==
               convT2d_out_at b cin h_in w_in cout kh kw
                              sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias tid))

val convt2d_general_alloc_f32 : convt2d_general_alloc_ty

inline_for_extraction let () = ()
