module Kuiper.KB.MatmulScaleResidual

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

(* Bridges between flat array2 ownership and its zero-cost rank-2 tensor view,
   required to feed the (tensor-based) GEMM API while keeping the rest of the
   proof array2-based. *)
ghost
fn bridge_fwd
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (#f : perm) (#s : erased (EM.chest2 et rows cols))
  requires on gpu_loc (a |-> Frac f s)
  ensures  on gpu_loc (a |-> Frac f s)
{
  rewrite (on gpu_loc (a |-> Frac f s))
       as (on gpu_loc (a |-> Frac f s));
}

ghost
fn bridge_bwd
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (#f : perm) (#s : erased (EM.chest2 et rows cols))
  requires on gpu_loc (a |-> Frac f s)
  ensures  on gpu_loc (a |-> Frac f s)
{
  rewrite (on gpu_loc (a |-> Frac f s))
       as (on gpu_loc (a |-> Frac f s));
}

(* Per-(i,j) discharge of [matmul_scale_residual_post].  Given the EXACT
   bias-add per-element fact for the flat index [i*out+j], the pointwise
   [scale_residual sf] map yields the desired float expression.
   [bias_add_at_ij] rewrites the flat bias-add entry into [chest2] form. *)
#push-options "--z3rlimit 50"
let msr_row_aux
  (batch out : nat)
  (sf : f32)
  (mm : EM.chest2 f32 batch out)
  (sbias : chest1 f32 out)
  (sy_b : chest1 f32 (batch * out))
  (hyp : squash
    (forall (tid:nat). tid < batch * out ==>
       acc1 sy_b tid == BA.bias_add_at batch out mm sbias tid))
  (i : natlt batch) (j : natlt out)
  : Lemma
      (ensures
        acc1 (chest_map (scale_residual sf) sy_b) (i * out + j) ==
          scale_residual sf (add (acc2 mm i j) (acc1 sbias j)))
  = BA.bias_add_at_ij batch out mm sbias i j;
    let tid : nat = i * out + j in
    assert (acc1 (chest_map (scale_residual sf) sy_b) tid
            == scale_residual sf (acc1 sy_b tid))
#pop-options

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn matmul_scale_residual_f32_impl
  (batch : szp)
  (input : szp)
  (out : szp {
     SZ.v batch * SZ.v out <= max_blocks * max_threads /\
     SZ.fits (SZ.v batch * SZ.v input) /\
     SZ.fits (SZ.v input * SZ.v out) /\
     SZ.fits (SZ.v batch * SZ.v out) })
  (sf : f32)
  (x    : array2 f32 (l2_row_major (SZ.v batch) (SZ.v input)) { is_global x    })
  (wt   : array2 f32 (l2_row_major (SZ.v input) (SZ.v out))   { is_global wt   })
  (bias : array1 f32 (l1_forward (SZ.v out))                  { is_global bias })
  (y    : array1 f32 (l1_forward (SZ.v batch * SZ.v out))     { is_global y    })
  (#sx   : erased (EM.chest2 f32 (SZ.v batch) (SZ.v input)))
  (#swt  : erased (EM.chest2 f32 (SZ.v input) (SZ.v out)))
  (#sbias: erased (chest1 f32 (SZ.v out)))
  (#sy   : erased (chest1 f32 (SZ.v batch * SZ.v out)))
  preserves cpu
  requires
    on gpu_loc (x    |-> sx)   **
    on gpu_loc (wt   |-> swt)  **
    on gpu_loc (bias |-> sbias)**
    on gpu_loc (y    |-> sy)
  ensures
    on gpu_loc (x    |-> sx)   **
    on gpu_loc (wt   |-> swt)  **
    on gpu_loc (bias |-> sbias)**
    (exists* (sy' : chest1 f32 (SZ.v batch * SZ.v out)).
       on gpu_loc (y |-> sy') **
       pure (matmul_scale_residual_post sf sx swt sbias sy'))
{
  (* Expose Seq.length / SZ.fits facts for the concrete layouts. *)

  (* Scratch GEMM output: gC = x @ wt  (batch × out). *)
  let gC = alloc0 #f32 (batch *^ out) (l2_row_major (SZ.v batch) (SZ.v out));
  with sc0. assert on gpu_loc (gC |-> sc0);
  bridge_fwd x;
  bridge_fwd wt;
  bridge_fwd gC;

  (* Launch 1: exact GEMM.  comb2 ignores the old gC value, so the result is
     exactly [matmul sx swt] (matmul_is_gemm SMTPat). *)
  P.mmcomb_gpu_exact (MS.comb2 #f32)
    #batch #out #input
    (x) (wt) (gC);
  with eC'. assert on gpu_loc (gC |-> eC');
  bridge_bwd x;
  bridge_bwd wt;
  bridge_bwd gC;
  assert pure (reveal eC' == MS.matmul (reveal sx) (reveal swt));

  let mm : erased (EM.chest2 f32 (SZ.v batch) (SZ.v out)) =
    hide (MS.matmul (reveal sx) (reveal swt));
  assert pure (eC' == mm);

  (* Launch 2: broadcast bias-add, writing the flat output y. *)
  BA.bias_add_gpu batch out gC bias y;
  with sy_b. assert on gpu_loc (y |-> sy_b);
  assert pure (forall (tid:nat). tid < SZ.v batch * SZ.v out ==>
                 acc1 sy_b tid == BA.bias_add_at (SZ.v batch) (SZ.v out) (reveal mm) sbias tid);

  (* Launch 3: pointwise scale-and-residual. *)
  Map.map_gpu (scale_residual sf) (batch *^ out) #_ #(c_l1_forward _) y;
  with sy'. assert on gpu_loc (y |-> sy');

  (* Discharge per-(i,j) [matmul_scale_residual_post]. *)
  Classical.forall_intro_2
    (msr_row_aux (SZ.v batch) (SZ.v out) sf (reveal mm) sbias sy_b ());

  free gC;
  ()
}
#pop-options

let matmul_scale_residual_f32 : matmul_scale_residual_ty f32 = matmul_scale_residual_f32_impl
