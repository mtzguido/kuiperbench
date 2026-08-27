module Kuiper.KB.Conv2DSquare

(* KernelBench L1 #63: square 2D convolution forward (bias=False).

   Wraps [Kuiper.Kernel.Conv2D.Naive.conv2d_naive_gpu] for the
   square-kernel, stride=1, pad=0, dilation=1 case (which #63 exercises
   with B=16, Cin=16, Cout=128, K=3, H=W=1024).

   Internally allocates a zero bias of length [cout] before calling the
   underlying primitive (the primitive itself takes a bias array; we
   pass a runtime-zero bias so the result is the bias-free conv).

   The spec connection back to [Kuiper.Spec.Conv2D.conv2d_out_at]
   inherits the admitted debt of the underlying primitive and is
   documented in the module's [skeptic.txt]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Kernel.Conv2D.Naive
module SZ = Kuiper.SizeT

(* Verified, extractable valid-conv output dimension: out = in - k + 1
   (stride 1, no padding, dilation 1).  Computed inside the verification
   boundary so the bridge does not derive the output shape in unverified C. *)
val conv2d_square_out_sz
  (l k : szp { SZ.v k <= SZ.v l })
  : Pure SZ.t (requires True) (ensures fun r -> SZ.v r == SZ.v l - SZ.v k + 1)

inline_for_extraction noextract
type conv2d_square_ty (t:Type0) {| scalar t |} =
  fn (b cin h_in cout k : szp)
     (h_out : szp { SZ.v h_out == SZ.v h_in - SZ.v k + 1 /\
                    conv2d_size_req b cin h_in h_in cout k k 1 h_out h_out })
     (gx : array1 t (l1_forward (b * cin * h_in * h_in))
           { is_global gx })
     (gw : array1 t (l1_forward (cout * cin * k * k))
           { is_global gw })
     (gbias : array1 t (l1_forward cout)
           { is_global gbias })
     (gy : array1 t (l1_forward (b * cout * h_out * h_out))
           { is_global gy })
     (#fx #fw #fb : perm)
     (#sx : erased (chest1 t (b * cin * h_in * h_in)))
     (#sw : erased (chest1 t (cout * cin * k * k)))
     (#sbias : erased (chest1 t cout))
     (#sy0 : erased (chest1 t (b * cout * h_out * h_out)))
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
       (exists* (sy : chest1 t (b * cout * h_out * h_out)).
         on gpu_loc (gy |-> sy) **
         pure (forall (tid : nat{tid < b * cout * h_out * h_out}).
                 acc1 sy tid ==
                 conv2d_out_at b cin h_in h_in cout k k 1 0 h_out h_out
                               sx sw sbias tid))

val conv2d_square_f32 : conv2d_square_ty f32

inline_for_extraction let () = ()
