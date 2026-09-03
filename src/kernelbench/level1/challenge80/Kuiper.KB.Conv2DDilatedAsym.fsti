module Kuiper.KB.Conv2DDilatedAsym

(* KernelBench L1 #80: 2D convolution with square input, asymmetric
   kernel, asymmetric padding, asymmetric dilation, single stride.
   The public fixed-challenge entry [conv2d_dilated_asym80_alloc_f32] derives
   both output dimensions, creates the zero bias and output, and wraps
   [Kuiper.Kernel.Conv2D.Dilated.conv2d_dilated_gpu].  The bridge only checks
   the fixed configuration and makes this one call. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2DDilated
open Kuiper.Kernel.Conv2D.Dilated
open Kuiper.Spec.Pool1D { pool_out_len_1d }
module SZ = Kuiper.SizeT

(* Verified, extractable dilated-conv output-dimension formula (see .fst). *)
val conv2dd_out_dim_sz
  (l k s d : szp) (p : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)

fn conv2d_dilated_asym_f32
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw : sz)
  (dh dw h_out : szp)
  (w_out : szp { conv2dd_size_req b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                  h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
        { is_global gy })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw_ : chest1 f32 (cout * cin * kh * kw))
  (#sbias : chest1 f32 cout)
  (#sy0 : chest1 f32 (b * cout * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
              acc1 sy tid ==
              conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                             h_out w_out sx sw_ sbias tid))

(* Complete fixed #80 entry: accepts only the original input and weight,
   constructs the bias=False value, allocates, and returns the final output. *)
fn conv2d_dilated_asym80_alloc_f32
  (gx : array1 f32 (l1_forward (8 * 32 * 512 * 512)) { is_global gx })
  (gw : array1 f32 (l1_forward (64 * 32 * 5 * 9)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (8 * 32 * 512 * 512))
  (#sw : chest1 f32 (64 * 32 * 5 * 9))
  preserves cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns gy : array1 f32 (l1_forward (8 * 64 * 508 * 496))
  ensures exists* (sy : chest1 f32 (8 * 64 * 508 * 496)).
    on gpu_loc (gy |-> sy) **
    pure (forall (tid : nat{tid < 8 * 64 * 508 * 496}).
      acc1 sy tid == conv2dd_out_at 8 32 512 512 64 5 9
        1 1 2 4 2 3 508 496 sx sw (mk1 (fun _ -> (zero #f32))) tid)


inline_for_extraction let () = ()
