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
inline_for_extraction noextract
type conv2d_general_ty (t:Type0) {| scalar t |} =
  fn (b cin h_in w_in cout kh kw stride : szp)
     (pad : sz)
     (h_out : szp)
     (w_out : szp { conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out })
     (gx : array1 t (l1_forward (b * cin * h_in * w_in))
           { is_global gx })
     (gw : array1 t (l1_forward (cout * cin * kh * kw))
           { is_global gw })
     (gbias : array1 t (l1_forward cout)
           { is_global gbias })
     (gy : array1 t (l1_forward (b * cout * h_out * w_out))
           { is_global gy })
     (#fx #fw #fb : perm)
     (#sx : erased (chest1 t (b * cin * h_in * w_in)))
     (#sw : erased (chest1 t (cout * cin * kh * kw)))
     (#sbias : erased (chest1 t cout))
     (#sy0 : erased (chest1 t (b * cout * h_out * w_out)))
     requires
       cpu **
       on gpu_loc (gx |-> Frac fx sx) **
       on gpu_loc (gw |-> Frac fw sw) **
       on gpu_loc (gbias |-> Frac fb sbias) **
       on gpu_loc (gy |-> sy0)
     ensures
       cpu **
       on gpu_loc (gx |-> Frac fx sx) **
       on gpu_loc (gw |-> Frac fw sw) **
       on gpu_loc (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 t (b * cout * h_out * w_out)).
         on gpu_loc (gy |-> sy) **
         pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
                 acc1 sy tid ==
                 conv2d_out_at b cin h_in w_in cout kh kw stride pad
                               h_out w_out sx sw sbias tid))

val conv2d_general_f32 : conv2d_general_ty f32

inline_for_extraction let () = ()
