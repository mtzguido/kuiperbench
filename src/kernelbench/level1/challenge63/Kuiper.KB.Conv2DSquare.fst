module Kuiper.KB.Conv2DSquare

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Kernel.Conv2D.Naive
module SZ = Kuiper.SizeT

(* Verified, extractable valid-conv output dimension: out = in - k + 1. *)
let conv2d_square_out_sz
  (l k : szp { SZ.v k <= SZ.v l })
  : Pure SZ.t (requires True) (ensures fun r -> SZ.v r == SZ.v l - SZ.v k + 1)
  = SZ.((l -^ k) +^ 1sz)

inline_for_extraction noextract
fn conv2d_square_impl
  (#et : Type0) {| scalar et |}
  (b cin h_in cout k : szp)
  (h_out : szp { SZ.v h_out == SZ.v h_in - SZ.v k + 1 /\
                 conv2d_size_req b cin h_in h_in cout k k 1 h_out h_out })
  (gx : array1 et (l1_forward (b * cin * h_in * h_in))
        { is_global gx })
  (gw : array1 et (l1_forward (cout * cin * k * k))
        { is_global gw })
  (gbias : array1 et (l1_forward cout)
        { is_global gbias })
  (gy : array1 et (l1_forward (b * cout * h_out * h_out))
        { is_global gy })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 et (b * cin * h_in * h_in))
  (#sw : chest1 et (cout * cin * k * k))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b * cout * h_out * h_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 et (b * cout * h_out * h_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * h_out * h_out}).
               acc1 sy tid ==
               conv2d_out_at b cin h_in h_in cout k k 1 0 h_out h_out
                             sx sw sbias tid))
{
  conv2d_naive_gpu #et b cin h_in h_in cout k k 1sz 0sz h_out h_out
                   gx gw gbias gy;
  ()
}

let conv2d_square_f32 = conv2d_square_impl #f32

inline_for_extraction let () = ()
