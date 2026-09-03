module Kuiper.KB.Conv2DSquare

(* KernelBench L1 #63: square 2D convolution forward (bias=False).

   Wraps [Kuiper.Kernel.Conv2D.Naive.conv2d_naive_gpu] for the
   square-kernel, stride=1, pad=0, dilation=1 case (which #63 exercises
   with B=16, Cin=16, Cout=128, K=3, H=W=1024).

   Internally allocates a zero bias of length [cout] before calling the
   underlying primitive (the primitive itself takes a bias array; we
   pass a runtime-zero bias so the result is the bias-free conv).

   The public [conv2d_square63_alloc_f32] entry derives the output extent,
   creates the zero bias and output, and calls the fully proved Conv2D
   primitive under the same [Kuiper.Spec.Conv2D.conv2d_out_at] functional
   postcondition. *)

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

fn conv2d_square_f32
  (b cin h_in cout k : szp)
  (h_out : szp { SZ.v h_out == SZ.v h_in - SZ.v k + 1 /\
                 conv2d_size_req b cin h_in h_in cout k k 1 h_out h_out })
  (gx : array1 f32 (l1_forward (b * cin * h_in * h_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * k * k))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (gy : array1 f32 (l1_forward (b * cout * h_out * h_out))
        { is_global gy })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * h_in))
  (#sw : chest1 f32 (cout * cin * k * k))
  (#sbias : chest1 f32 cout)
  (#sy0 : chest1 f32 (b * cout * h_out * h_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 f32 (b * cout * h_out * h_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b * cout * h_out * h_out}).
              acc1 sy tid ==
              conv2d_out_at b cin h_in h_in cout k k 1 0 h_out h_out
                            sx sw sbias tid))

(* Complete fixed #63 operation: input/weight are the original PyTorch
   buffers; zero bias, output geometry, allocation, and launch are internal. *)
fn conv2d_square63_alloc_f32
  (gx : array1 f32 (l1_forward (16 * 16 * 1024 * 1024)) { is_global gx })
  (gw : array1 f32 (l1_forward (128 * 16 * 3 * 3)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (16 * 16 * 1024 * 1024))
  (#sw : chest1 f32 (128 * 16 * 3 * 3))
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns gy : array1 f32 (l1_forward (16 * 128 * 1022 * 1022))
  ensures exists* (sy : chest1 f32 (16 * 128 * 1022 * 1022)).
    on gpu_loc (gy |-> sy) **
    pure (forall (tid : nat{tid < 16 * 128 * 1022 * 1022}).
      acc1 sy tid == conv2d_out_at 16 16 1024 1024 128 3 3 1 0
        1022 1022 sx sw (mk1 (fun _ -> (zero #f32))) tid)


inline_for_extraction let () = ()
