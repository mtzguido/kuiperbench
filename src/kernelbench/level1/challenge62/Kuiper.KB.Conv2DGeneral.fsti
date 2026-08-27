module Kuiper.KB.Conv2DGeneral

(* Generic 2D-convolution-forward entry point used by KernelBench L1
   #50, #55, #56, #62.  Exposes every parameter that the underlying
   [Kuiper.Kernel.Conv2D.Naive.conv2d_naive_gpu] supports: asymmetric
   inputs (h_in, w_in), asymmetric kernels (kh, kw), non-unit stride,
   non-zero padding.  Caller passes the bias array directly. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Kernel.Conv2D.Naive
fn conv2d_general_f32
  (b cin h_in w_in cout kh kw stride : szp)
  (pad : sz)
  (h_out : szp)
  (w_out : szp { conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out })
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
  (#sw : chest1 f32 (cout * cin * kh * kw))
  (#sbias : chest1 f32 cout)
  (#sy0 : chest1 f32 (b * cout * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
              acc1 sy tid ==
              conv2d_out_at b cin h_in w_in cout kh kw stride pad
                            h_out w_out sx sw sbias tid))


inline_for_extraction let () = ()
