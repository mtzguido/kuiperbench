module Kuiper.KB.ConvT2DGrouped75

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Kernel.ConvT2D.GroupedNaive
module Map = Kuiper.Kernel.Map
module Chest = Kuiper.Chest

inline_for_extraction noextract
let const_zero_f32 (_ : f32) : f32 = zero

let zero_bias : chest1 f32 64 = mk1 (fun _ -> zero)

let map_const_zero (#n : nat) (s : chest1 f32 n)
  : Lemma (chest_map const_zero_f32 s == mk1 (fun _ -> zero))
  = Chest.lemma_equal_intro
      (chest_map const_zero_f32 s) (mk1 (fun _ -> zero));
    Chest.ext (chest_map const_zero_f32 s) (mk1 (fun _ -> zero))

inline_for_extraction noextract
fn convt2d_grouped75_alloc
  (gx : array1 f32 (l1_forward (16 * 32 * 128 * 256))
        { is_global gx })
  (gw : array1 f32 (l1_forward (32 * 16 * 3 * 5))
        { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (16 * 32 * 128 * 256))
  (#sw : chest1 f32 (32 * 16 * 3 * 5))
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns gy : array1 f32 (l1_forward (16 * 64 * 257 * 766))
  ensures
    exists* (sy : chest1 f32 (16 * 64 * 257 * 766)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < 16 * 64 * 257 * 766}).
        acc1 sy tid == convt2d_grouped75_out_at sx sw tid)
{
  let gbias = alloc0 #f32 64sz (l1_forward 64sz);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 64sz gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> zero_bias));

  let gy = alloc0 #f32 (16sz *^ 64sz *^ 257sz *^ 766sz)
                       (l1_forward (16sz *^ 64sz *^ 257sz *^ 766sz));
  with ey. assert (on gpu_loc (gy |-> ey));
  convt2d_grouped_naive_gpu
    16sz 8sz 128sz 256sz 16sz 3sz 5sz 2sz 3sz 1sz 2sz 2sz 1sz 257sz 766sz
    gx gw gbias gy;
  free gbias;
  gy
}

let convt2d_grouped75_alloc_f32 = convt2d_grouped75_alloc

inline_for_extraction let () = ()
