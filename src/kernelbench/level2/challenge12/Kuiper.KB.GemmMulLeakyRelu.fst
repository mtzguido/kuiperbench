module Kuiper.KB.GemmMulLeakyRelu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, c_l1_forward, c_l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module KS = Kuiper.Seq.Common
module Map = Kuiper.Kernel.Map
module BA = Kuiper.Kernel.BiasAdd
module P = Kuiper.Kernel.GEMM.Naive2

(* Per-(i,j) discharge of [gemm_mul_lrelu_post].  Given the EXACT bias-add
   per-element fact for the flat index [i*out+j], the fused pointwise
   [mul_lrelu mult slope] map yields the desired float expression.
   [bias_add_at_ij] rewrites the flat bias-add entry into [chest2] form.

   Wrapped in [--z3rlimit 50] because the [lseq_map] ensures indexes at
   [i*out+j], which needs [i*out+j < batch*out] (otherwise F* fails with a
   subtyping check, "got type int"). *)

(* Generic [chest_map]/[acc1] commutation, kept with the map function [f]
   opaque.  [mul_lrelu] is a concrete [let] with a conditional and nonlinear
   [mul]; letting Z3 unfold it under [chest_map] destabilises the query, so we
   discharge the aux with [--fuel 0] and feed this pre-proved fact instead. *)
let acc1_chest_map
  (#et1 #et2 : Type) (#n : nat)
  (f : et1 -> et2) (c : chest1 et1 n) (i : natlt n)
  : Lemma (acc1 (chest_map f c) i == f (acc1 c i))
  = ()

#push-options "--z3rlimit 50 --fuel 0 --ifuel 0"
let gmlr_row_aux
  (batch out : nat)
  (mult slope : f32)
  (mm : chest2 f32 batch out)
  (sbias : chest1 f32 out)
  (sy_b : chest1 f32 (batch * out))
  (hyp : squash
    (forall (tid:nat). tid < batch * out ==>
       acc1 sy_b tid == BA.bias_add_at batch out mm sbias tid))
  (i : natlt batch) (j : natlt out)
  : Lemma
      (ensures
        acc1 (chest_map (mul_lrelu mult slope) sy_b) (i * out + j) ==
          mul_lrelu mult slope (add (acc2 mm i j) (acc1 sbias j)))
  = BA.bias_add_at_ij batch out mm sbias i j;
    acc1_chest_map (mul_lrelu mult slope) sy_b (i * out + j)
#pop-options

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn gemm_mul_leaky_relu_f32_impl
  (batch : szp)
  (input : szp)
  (out : szp {
     SZ.v batch * SZ.v out <= max_blocks * max_threads /\
     SZ.fits (SZ.v batch * SZ.v input) /\
     SZ.fits (SZ.v input * SZ.v out) /\
     SZ.fits (SZ.v batch * SZ.v out) })
  (mult : f32)
  (slope : f32)
  (x    : array2 f32 (l2_row_major batch input) { is_global x    })
  (wt   : array2 f32 (l2_row_major input out)   { is_global wt   })
  (bias : array1 f32 (l1_forward out)                  { is_global bias })
  (y    : array1 f32 (l1_forward (SZ.v batch * SZ.v out))     { is_global y    })
  (#sx   : chest2 f32 batch input)
  (#swt  : chest2 f32 input out)
  (#sbias: chest1 f32 out)
  (#sy   : chest1 f32 (SZ.v batch * SZ.v out))
  preserves
    cpu **
    on gpu_loc (x    |-> sx) **
    on gpu_loc (wt   |-> swt) **
    on gpu_loc (bias |-> sbias)
  requires
    on gpu_loc (y    |-> sy)
  ensures
    (exists* (sy' : chest1 f32 (SZ.v batch * SZ.v out)).
       on gpu_loc (y |-> sy') **
       pure (gemm_mul_lrelu_post mult slope sx swt sbias sy'))
{
  (* Expose Seq.length / SZ.fits facts for the concrete layouts. *)

  (* Scratch GEMM output: gC = x @ wt  (batch × out). *)
  let gC = alloc0 #f32 (batch *^ out) (l2_row_major batch out);
  with sc0. assert on gpu_loc (gC |-> sc0);

  (* Launch 1: exact GEMM.  comb2 ignores the old gC value. *)
  P.mmcomb_gpu_exact (MS.comb2 #f32)
    #batch #out #input
    (x) (wt) (gC);
  with eC'. assert on gpu_loc (gC |-> eC');
  assert pure (reveal eC' == MS.matmul (reveal sx) (reveal swt));

  let mm : chest2 f32 batch out =
    hide (MS.matmul (reveal sx) (reveal swt));
  assert pure (eC' == mm);

  (* Launch 2: broadcast bias-add, writing the flat output y. *)
  BA.bias_add_gpu batch out gC bias y;
  with sy_b. assert on gpu_loc (y |-> sy_b);
  assert pure (forall (tid:nat). tid < SZ.v batch * SZ.v out ==>
                 acc1 sy_b tid == BA.bias_add_at batch out (reveal mm) sbias tid);

  (* Launch 3: fused pointwise multiply-by-constant then LeakyReLU. *)
  Map.map_gpu (mul_lrelu mult slope) (batch *^ out) #_ #(c_l1_forward _) y;
  with sy'. assert on gpu_loc (y |-> sy');

  (* Discharge per-(i,j) [gemm_mul_lrelu_post]. *)
  Classical.forall_intro_2
    (gmlr_row_aux batch out mult slope (reveal mm) sbias sy_b ());

  free gC;
  ()
}
#pop-options

let gemm_mul_leaky_relu_f32 = gemm_mul_leaky_relu_f32_impl
