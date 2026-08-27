module Kuiper.KB.Conv1DGeneral

(* Generic 1D-convolution-forward entry point used by KernelBench L1
   #67 and #76.  Exposes every parameter that the underlying
   [Kuiper.Kernel.Conv1D.Naive.conv1d_naive_gpu] supports: stride>=1,
   pad>=0, dilation>=1.  Caller passes the bias array directly. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv1D
open Kuiper.Kernel.Conv1D.Naive
inline_for_extraction noextract
type conv1d_general_ty (t:Type0) {| scalar t |} =
  fn (b cin l_in cout kk stride : szp)
     (pad : sz)
     (dilation : szp)
     (l_out : szp { conv1d_size_req b cin l_in cout kk stride dilation l_out })
     (gx : array1 t (l1_forward (b * cin * l_in))
           { is_global gx })
     (gw : array1 t (l1_forward (cout * cin * kk))
           { is_global gw })
     (gbias : array1 t (l1_forward cout)
           { is_global gbias })
     (gy : array1 t (l1_forward (b * cout * l_out))
           { is_global gy })
     (#fx #fw #fb : perm)
     (#sx : chest1 t (b * cin * l_in))
     (#sw : chest1 t (cout * cin * kk))
     (#sbias : chest1 t cout)
     (#sy0 : chest1 t (b * cout * l_out))
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
       (exists* (sy : chest1 t (b * cout * l_out)).
         on gpu_loc (gy |-> sy) **
         pure (forall (tid : nat{tid < b * cout * l_out}).
                 acc1 sy tid ==
                 conv1d_out_at b cin l_in cout kk stride pad dilation
                               l_out sx sw sbias tid))

val conv1d_general_f32 : conv1d_general_ty f32

inline_for_extraction let () = ()
