module Kuiper.KB.ConvT3DGrouped72

(* Complete verified entry point for the fixed grouped ConvTranspose3D in
   KernelBench L1 #72. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Kernel.ConvT3D.GroupedNaive

let convt3d_grouped72_out_at
  (sx : chest1 f32 (8 * 32 * 12 * 24 * 48))
  (sw : chest1 f32 (32 * 8 * 3 * 5 * 7))
  (tid : nat{tid < 8 * 32 * 24 * 48 * 96})
  : GTot f32
  = convT3d_grouped_out_at
      8 8 12 24 48 8 3 5 7 2 2 2 1 2 3 1 1 1 24 48 96
      sx sw (mk1 (fun _ -> zero)) tid

fn convt3d_grouped72_alloc_f32
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

inline_for_extraction let () = ()
