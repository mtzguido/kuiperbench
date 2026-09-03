module Kuiper.KB.Conv2DSquare

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Kernel.Conv2D.Naive
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map
module Chest = Kuiper.Chest

inline_for_extraction noextract
let const_zero_f32 (_ : f32) : f32 = zero

let map_const_zero (#n : nat) (s : chest1 f32 n)
  : Lemma (chest_map const_zero_f32 s == mk1 (fun _ -> zero))
  = Chest.lemma_equal_intro
      (chest_map const_zero_f32 s) (mk1 (fun _ -> zero));
    Chest.ext (chest_map const_zero_f32 s) (mk1 (fun _ -> zero))

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
{
  let gbias = alloc0 #f32 128sz (l1_forward 128sz);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 128sz gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> mk1 (fun _ -> (zero #f32))));

  let gy = alloc0 #f32 (16sz *^ 128sz *^ 1022sz *^ 1022sz)
    (l1_forward (16sz *^ 128sz *^ 1022sz *^ 1022sz));
  with ey. assert (on gpu_loc (gy |-> ey));
  conv2d_square_f32 16sz 16sz 1024sz 128sz 3sz 1022sz
    gx gw gbias gy;
  free gbias;
  gy
}

inline_for_extraction let () = ()
