module Kuiper.KB.Conv2DDilatedAsym

(* KernelBench L1 #80: 2D convolution with square input, asymmetric
   kernel, asymmetric padding, asymmetric dilation, single stride.
   Wraps [Kuiper.Kernel.Conv2D.Dilated.conv2d_dilated_gpu] with all
   per-axis parameters exposed to the bridge.  Caller passes the
   bias array directly (zeroed scratch in the bias=False case). *)

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
         SZ.v r == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d))

inline_for_extraction noextract
type conv2d_dilated_asym_ty (t:Type0) {| scalar t |} =
  fn (b cin h_in w_in cout kh kw sh sw : szp)
     (ph pw : sz)
     (dh dw h_out : szp)
     (w_out : szp { conv2dd_size_req b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                     h_out w_out })
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
     (#sw_ : erased (chest1 t (cout * cin * kh * kw)))
     (#sbias : erased (chest1 t cout))
     (#sy0 : erased (chest1 t (b * cout * h_out * w_out)))
     requires
       cpu **
       on gpu_loc (gx |-> Frac fx sx) **
       on gpu_loc (gw |-> Frac fw sw_) **
       on gpu_loc (gbias |-> Frac fb sbias) **
       on gpu_loc (gy |-> sy0)
     ensures
       cpu **
       on gpu_loc (gx |-> Frac fx sx) **
       on gpu_loc (gw |-> Frac fw sw_) **
       on gpu_loc (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 t (b * cout * h_out * w_out)).
         on gpu_loc (gy |-> sy) **
         pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
                 acc1 sy tid ==
                 conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                h_out w_out sx sw_ sbias tid))

val conv2d_dilated_asym_f32 : conv2d_dilated_asym_ty f32

inline_for_extraction let () = ()
