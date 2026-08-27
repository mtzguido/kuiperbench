module Kuiper.KB.Conv3DGeneral

(* Generic 3D-convolution-forward entry point used by KernelBench L1
   #54, #59, #60, #66.  Exposes every parameter that the underlying
   [Kuiper.Kernel.Conv3D.Naive.conv3d_naive_gpu] supports: asymmetric
   inputs (d_in, h_in, w_in), asymmetric kernels (kd, kh, kw),
   non-unit stride, non-zero padding (single pad/stride value, since
   all four upstream KernelBench tests default to symmetric).  Caller
   passes the bias array directly. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Kernel.Conv3D.Naive
fn conv3d_general_f32
  (b cin d_in h_in w_in cout kd kh kw stride : szp)
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
  (gy : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out))
        { is_global gy })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  (#sy0 : chest1 f32 (b * cout * d_out * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 f32 (b * cout * d_out * h_out * w_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
              acc1 sy tid ==
              conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                            d_out h_out w_out sx sw sbias tid))


inline_for_extraction let () = ()
