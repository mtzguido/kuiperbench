module Kuiper.KB.GemmDivSumScale

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, c_l1_forward, c_l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module Map = Kuiper.Kernel.Map
module KS = Kuiper.Seq.Common
module HRedB = Kuiper.Kernel.HReduce.Block
module P = Kuiper.Kernel.GEMM.Naive2
module PApprox = Kuiper.Kernel.GEMM.Naive3

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

(* [lseq_map id s == s], hence the two rsums agree. *)
let row_simpl
  (#rows #cols : nat)
  (vr : EM.chest2 real rows cols)
  (r : natlt rows)
  : Lemma (rsum (KS.lseq_map id (EM.ematrix_row vr r))
           == rsum (EM.ematrix_row vr r))
  = Seq.lemma_eq_intro
      (KS.lseq_map id (EM.ematrix_row vr r))
      (EM.ematrix_row vr r)

(* Per-row discharge: if [sout @! r] approximates the (id-mapped) row-sum,
   then after the scalar multiply [mul _ k] the entry approximates the
   row-sum scaled by [to_real k].  Uses the [a_mul_pat] SMTPat from
   Kuiper.Approximates.Base. *)
let final_row_aux
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (vr : EM.chest2 real rows cols)
  (k : t)
  (sout : chest1 t rows)
  (r : natlt rows)
  : Lemma
      (requires (acc1 sout r) %~ rsum (KS.lseq_map id (EM.ematrix_row vr r)))
      (ensures
        (acc1 (chest_map (mul k) sout) r)
         %~ (rsum (EM.ematrix_row vr r) *. to_real k))
  = row_simpl vr r;
    assert (acc1 (chest_map (mul k) sout) r == mul k (acc1 sout r));
    assert (k %~ to_real k)

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn gemm_div_sum_scale_f32_impl
  (batch : szp)
  (input : szp)
  (hidden : szp {
     SZ.v batch <= max_blocks /\
     SZ.v batch * SZ.v hidden <= max_blocks * max_threads /\
     SZ.fits (SZ.v hidden + max_threads) /\
     SZ.fits (SZ.v batch * SZ.v input) /\
     SZ.fits (SZ.v input * SZ.v hidden) /\
     SZ.fits (SZ.v batch * SZ.v hidden) })
  (k : f32)
  (x  : array2 f32 (l2_row_major (SZ.v batch) (SZ.v input))  { is_global x  })
  (wt : array2 f32 (l2_row_major (SZ.v input) (SZ.v hidden)) { is_global wt })
  (y  : array1 f32 (l1_forward (SZ.v batch))                 { is_global y  })
  (#sx  : erased (EM.chest2 f32 (SZ.v batch) (SZ.v input)))
  (#swt : erased (EM.chest2 f32 (SZ.v input) (SZ.v hidden)))
  (#sy  : erased (chest1 f32 (SZ.v batch)))
  preserves cpu
  requires
    on gpu_loc (x  |-> sx)  **
    on gpu_loc (wt |-> swt) **
    on gpu_loc (y  |-> sy)
  ensures
    on gpu_loc (x  |-> sx)  **
    on gpu_loc (wt |-> swt) **
    (exists* (sy' : chest1 f32 (SZ.v batch)).
       on gpu_loc (y |-> sy') **
       pure (gdss_post k sx swt sy'))
{
  (* Expose Seq.length / SZ.fits facts for the concrete layouts. *)

  (* Scratch output of the GEMM: gC = x @ wt  (batch × hidden). *)
  let gC = alloc0 #f32 (batch *^ hidden) (l2_row_major (SZ.v batch) (SZ.v hidden));
  with sc0. assert on gpu_loc (gC |-> sc0);

  (* Real witnesses for the approximate GEMM spec. *)
  let rA : erased (EM.chest2 real (SZ.v batch) (SZ.v input))  =
    hide (EM.to_real_matrix (reveal sx));
  let rB : erased (EM.chest2 real (SZ.v input) (SZ.v hidden)) =
    hide (EM.to_real_matrix (reveal swt));
  let rC : erased (EM.chest2 real (SZ.v batch) (SZ.v hidden)) =
    hide (EM.to_real_matrix (reveal sc0));

  assert pure (MS.comb2 #f32 `approx2` MS.comb2 #real);

  (* Bridge Array2 ownership into tensor ownership for the new GEMM API. *)
  bridge_fwd x;
  bridge_fwd wt;
  bridge_fwd gC;

  (* Launch 1: GEMM. comb2 ignores the old gC value, so the result is the
     plain real matmul (matmul_is_gemm SMTPat). *)
  PApprox.mmcomb_gpu_approx (MS.comb2 #f32) (MS.comb2 #real)
    #batch #hidden #input
    (x) (wt) (gC)
    (reveal rA) (reveal rB) (reveal rC);
  with eC'. assert on gpu_loc (gC |-> eC');
  bridge_bwd x;
  bridge_bwd wt;
  bridge_bwd gC;
  assert pure (eC' %~ MS.matmul (reveal rA) (reveal rB));

  (* Launch 2: per-row tree reduction (sum over hidden). cols = hidden is
     NOT capped at max_threads in reduce_batched_block; only nth (=1024) is. *)
  let vr : erased (EM.chest2 real (SZ.v batch) (SZ.v hidden)) =
    hide (MS.matmul (reveal rA) (reveal rB));
  HRedB.reduce_batched_block #f32 id id batch hidden 1024sz
    #_ #(c_l2_row_major (SZ.v batch) hidden)
    #_ #(c_l1_forward _)
    gC y vr;
  with sout. assert on gpu_loc (y |-> sout);

  (* Launch 3: scale every entry by k. *)
  assert pure (SZ.v batch <= max_blocks * max_threads);
  Map.map_gpu (mul k) batch #_ #(c_l1_forward _) y;

  (* Discharge per-row [gdss_post]. *)
  Classical.forall_intro
    (Classical.move_requires
       (final_row_aux #f32 (reveal vr) k sout));

  free gC;
  ()
}
#pop-options

let gemm_div_sum_scale_f32 : gemm_div_sum_scale_ty f32 = gemm_div_sum_scale_f32_impl
