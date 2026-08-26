module Kuiper.KB.RMSNorm

(* KernelBench L1 #36: RMS normalisation, 3 GPU launches.

   Launch 1: reduce_batched sq_step  → sum_sq[i]  = Σ_c x[i,c]²
   Launch 2: map_gpu inv_rms_fn      → sum_sq[i]  = rsqrt(sum_sq[i]/C+ε)
   Launch 3: row_scale sum_sq x      → x[i,c] ← x[i,c] * sum_sq[i]

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.RMSNorm
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map
module HRed = Kuiper.Kernel.HReduce
module RowScale = Kuiper.Kernel.RowScale
module KS = Kuiper.Seq.Common

(* post_map for inv-rms.  The pre_map for the reduction itself is
   [Kuiper.Spec.RMSNorm.sq_step], reused so that the spec lemmas
   apply unchanged. *)
inline_for_extraction noextract
let inv_rms_fn (eps inv_c : f32) (s : f32) : f32 =
  rsqrt (add (mul s inv_c) eps)

(* Per-row functional postcondition.  After the three GPU launches the
   array [x] holds, by composition of the three primitives' specs, the
   matrix
     s_row_scale (lseq_map (inv_rms_fn eps inv_c)
                            (seq_reduce_rows sq_step sx))
                 sx
   The witness for [row_rmsnormalized] at row [r] is then
     sumsq_r = row_reduce_partial sq_step sx r c
     inv_r   = inv_rms_fn eps inv_c sumsq_r
   The [sumsq_r %~ frobenius_sumsq_r ...] obligation reduces to
   [row_reduce_partial_sq_approx]. *)
#push-options ""
let rmsnorm_row_aux
  (bhw_n c_n : nat)
  (eps inv_c : f32)
  (sx : EM.chest2 f32 bhw_n c_n)
  (r : nat)
  : Lemma (requires r < bhw_n)
          (ensures  row_rmsnormalized eps inv_c sx
                      (Kuiper.Kernel.RowScale.s_row_scale
                         (chest_map (inv_rms_fn eps inv_c)
                            (seq_to_chest1 (HRed.seq_reduce_rows (sq_step #f32) sx)))
                         sx)
                      r)
  = row_reduce_partial_sq_approx sx r;
    let sfac : chest1 f32 bhw_n =
      chest_map (inv_rms_fn eps inv_c)
                (seq_to_chest1 (HRed.seq_reduce_rows (sq_step #f32) sx)) in
    let sumsq : f32 = HRed.row_reduce_partial (sq_step #f32) sx r c_n in
    (* acc1 sfac r reduces through chest_map / seq_to_chest1 / init_ghost
       to [inv_rms_fn eps inv_c (row_reduce_partial sq_step sx r c_n)]. *)
    assert (acc1 sfac r == inv_rms_fn eps inv_c sumsq)
#pop-options

let rmsnorm_post_aux
  (bhw_n c_n : nat)
  (eps inv_c : f32)
  (sx : EM.chest2 f32 bhw_n c_n)
  : Lemma (rmsnorm_post bhw_n c_n eps inv_c sx
             (Kuiper.Kernel.RowScale.s_row_scale
                (chest_map (inv_rms_fn eps inv_c)
                   (seq_to_chest1 (HRed.seq_reduce_rows (sq_step #f32) sx)))
                sx))
  = let aux (r : nat { r < bhw_n }) : Lemma
      (row_rmsnormalized eps inv_c sx
        (Kuiper.Kernel.RowScale.s_row_scale
          (chest_map (inv_rms_fn eps inv_c)
            (seq_to_chest1 (HRed.seq_reduce_rows (sq_step #f32) sx)))
          sx)
        r)
      = rmsnorm_row_aux bhw_n c_n eps inv_c sx r
    in
    Classical.forall_intro aux

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn rmsnorm_fw_f32_impl
  (b : szp)
  (hw : SZ.t { 0 < SZ.v hw /\ SZ.fits (SZ.v b * SZ.v hw) })
  (c : SZ.t { 0 < SZ.v c /\ SZ.fits (SZ.v hw * SZ.v c) /\ SZ.fits (SZ.v b * (SZ.v hw * SZ.v c)) })
  (eps : f32)
  (inv_c : f32)
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v hw) (SZ.v c)) { is_global x })
  (#sx : erased (EM.chest2 f32 (SZ.v b * SZ.v hw) (SZ.v c)))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b * SZ.v hw > 0 /\
      SZ.v b * SZ.v hw * SZ.v c <= max_blocks * max_threads
    )
  ensures
    exists* (sx' : EM.chest2 f32 (SZ.v b * SZ.v hw) (SZ.v c)).
      on gpu_loc (x |-> sx') **
      pure (rmsnorm_post (SZ.v b * SZ.v hw) (SZ.v c) eps inv_c sx sx')
{
  (* bhw = B * HW; strictly positive since b > 0 and hw > 0 *)
  let bhw : szp = b *^ hw;

  (* ── Launch 1: sum of squares per row ── *)
  let sum_sq = alloc0 #f32 bhw (l1_forward bhw);
  with em. assert (on gpu_loc (sum_sq |-> em));

  (* bhw <= max_blocks * max_threads  (since c >= 1 and bhw*c <= bound) *)
  assert pure (SZ.v bhw <= max_blocks * max_threads);
  HRed.reduce_batched (sq_step #f32) bhw c #_ #(c_l2_bcm_pages (SZ.v b) hw c) x sum_sq;

  (* ── Launch 2: inv_rms per row ── *)
  Map.map_gpu (inv_rms_fn eps inv_c) bhw sum_sq;

  (* ── Launch 3: in-place row scale ── *)
  (* Need bhw * c <= max_blocks * max_threads *)
  assert pure (SZ.v bhw * SZ.v c <= max_blocks * max_threads);
  RowScale.row_scale bhw c sum_sq #_ #(c_l2_bcm_pages (SZ.v b) hw c) x;

  (* Free scratch buffer; row_scale preserves sum_sq at full perm *)
  free sum_sq;

  (* Discharge the per-row functional postcondition. *)
  rmsnorm_post_aux (SZ.v bhw) (SZ.v c) eps inv_c sx;
  ()
}
#pop-options

(* Public entry point: compute the per-row reciprocal [rms_inv_c c] inside
   the verification boundary (extracts to 1.0f / (float)C), then delegate to
   [rmsnorm_fw_f32_impl].  The proof above treats [inv_c] abstractly, so this
   constant computation does not affect its cost. *)
fn rmsnorm_fw
  (b : szp)
  (hw : SZ.t { 0 < SZ.v hw /\ SZ.fits (SZ.v b * SZ.v hw) })
  (c : SZ.t { 0 < SZ.v c /\ SZ.fits (SZ.v hw * SZ.v c) /\ SZ.fits (SZ.v b * (SZ.v hw * SZ.v c)) })
  (eps : f32)
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v hw) (SZ.v c)) { is_global x })
  (#sx : erased (EM.chest2 f32 (SZ.v b * SZ.v hw) (SZ.v c)))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b * SZ.v hw > 0 /\
      SZ.v b * SZ.v hw * SZ.v c <= max_blocks * max_threads
    )
  ensures
    exists* (sx' : EM.chest2 f32 (SZ.v b * SZ.v hw) (SZ.v c)).
      on gpu_loc (x |-> sx') **
      pure (rmsnorm_post (SZ.v b * SZ.v hw) (SZ.v c) eps (rms_inv_c c) sx sx')
{
  let inv_c : f32 = rms_inv_c c;
  rmsnorm_fw_f32_impl b hw c eps inv_c x;
}

let rmsnorm_fw_f32 : rmsnorm_fw_ty f32 = rmsnorm_fw
