module Kuiper.KB.ConvT2DGrouped75

(* Complete verified entry point for the fixed grouped ConvTranspose2D in
   KernelBench L1 #75.  The public ABI takes the two original contiguous
   tensors and returns the complete NCHW result; zero-bias construction,
   output allocation, group selection, and grouped indexing stay below this
   boundary. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Kernel.ConvT2D.GroupedNaive

let convt2d_grouped75_out_at
  (sx : chest1 f32 (16 * 32 * 128 * 256))
  (sw : chest1 f32 (32 * 16 * 3 * 5))
  (tid : nat{tid < 16 * 64 * 257 * 766})
  : GTot f32
  = convT2d_grouped_out_at
      16 8 128 256 16 3 5 2 3 1 2 2 1 257 766
      sx sw (mk1 (fun _ -> zero)) tid

fn convt2d_grouped75_alloc_f32
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

inline_for_extraction let () = ()
