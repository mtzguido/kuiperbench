module Kuiper.KB.HingeLoss

(* KernelBench L1 #100: broadcast hinge loss.

   The public entry accepts the canonical inputs directly:
     predictions : (B, N), row-major
     targets     : (N,)

   It preserves both inputs, computes
     mean_{i,j} max(0, 1 - predictions[i,j] * targets[j])
   through verified Kuiper kernels, and returns a freshly allocated
   one-element GPU buffer whose ownership passes to the caller. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
open Kuiper.Spec.HingeLoss
module SZ = Kuiper.SizeT

fn hinge_loss_broadcast_f32
  (b : szp)
  (n : szp {
     SZ.v b * SZ.v n <= max_blocks * max_threads /\
     SZ.fits (SZ.v b * SZ.v n) /\
     SZ.fits (SZ.v b * SZ.v n + max_threads) })
  (predictions : array2 f32 (l2_row_major b n) { is_global predictions })
  (targets     : array1 f32 (l1_forward n) { is_global targets })
  (#sp : chest2 f32 b n)
  (#st : chest1 f32 n)
  (rp : erased (chest2 real b n))
  (rt : erased (chest1 real n))
  (#fp #ft : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (predictions |-> Frac fp sp) **
    on gpu_loc (targets |-> Frac ft st) **
    pure (sp %~ rp /\ st %~ rt)
  returns out : array1 f32 (l1_forward 1)
  ensures
    (exists* (sout : chest1 f32 1).
       on gpu_loc (out |-> sout) **
       pure (acc1 sout 0 %~ real_hinge_broadcast b n rp rt))
