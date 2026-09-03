module Kuiper.KB.ConvT3DGrouped72

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Kernel.ConvT3D.GroupedNaive
module Map = Kuiper.Kernel.Map
module Chest = Kuiper.Chest

inline_for_extraction noextract
let const_zero_f32 (_ : f32) : f32 = zero

let zero_bias : chest1 f32 32 = mk1 (fun _ -> zero)

let map_const_zero (#n : nat) (s : chest1 f32 n)
  : Lemma (chest_map const_zero_f32 s == mk1 (fun _ -> zero))
  = Chest.lemma_equal_intro
      (chest_map const_zero_f32 s) (mk1 (fun _ -> zero));
    Chest.ext (chest_map const_zero_f32 s) (mk1 (fun _ -> zero))

inline_for_extraction noextract
fn convt3d_grouped72_alloc
  (gx : array1 f32 (l1_forward (8 * 32 * 12 * 24 * 48))
        { is_global gx })
  (gw : array1 f32 (l1_forward (32 * 8 * 3 * 5 * 7))
        { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (8 * 32 * 12 * 24 * 48))
  (#sw : chest1 f32 (32 * 8 * 3 * 5 * 7))
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns gy : array1 f32 (l1_forward (8 * 32 * 24 * 48 * 96))
  ensures
    exists* (sy : chest1 f32 (8 * 32 * 24 * 48 * 96)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < 8 * 32 * 24 * 48 * 96}).
        acc1 sy tid == convt3d_grouped72_out_at sx sw tid)
{
  let gbias = alloc0 #f32 32sz (l1_forward 32sz);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 32sz gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> zero_bias));

  let gy = alloc0 #f32 (8sz *^ 32sz *^ 24sz *^ 48sz *^ 96sz)
                       (l1_forward (8sz *^ 32sz *^ 24sz *^ 48sz *^ 96sz));
  with ey. assert (on gpu_loc (gy |-> ey));
  convt3d_grouped_naive_gpu
    8sz 8sz 12sz 24sz 48sz 8sz 3sz 5sz 7sz
    2sz 2sz 2sz 1sz 2sz 3sz 1sz 1sz 1sz 24sz 48sz 96sz
    gx gw gbias gy;
  free gbias;
  gy
}

let convt3d_grouped72_alloc_f32 = convt3d_grouped72_alloc

inline_for_extraction let () = ()
