module Kuiper.KB.RMSNorm

(* KernelBench L1 #36: RMS normalisation, 3 GPU launches.

   Launch 1: reduce_batched sq_step  → sum_sq[i]  = Σ_c x[i,c]²
   Launch 2: map_gpu inv_rms_fn      → sum_sq[i]  = rsqrt(sum_sq[i]/C+ε)
   Launch 3: row_scale sum_sq x      → x[i,c] ← x[i,c] * sum_sq[i]

   The direct real proof uses the temporary [rsqrt_approx]
   compatibility assumption documented in the repository patch. *)

#lang-pulse
open Kuiper
open Kuiper.Float.Casts
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.RMSNorm
open Kuiper.Spec.Frobenius
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map
module HRed = Kuiper.Kernel.HReduce
module RowScale = Kuiper.Kernel.RowScale
module KS = Kuiper.Seq.Common
module RsqrtApprox = Kuiper.KB.Compat.RsqrtApprox
module Copy = Kuiper.KB.Tensor.Copy

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
   The proof relates the reduction and reciprocal-square-root intermediates
   to their real counterparts before proving the direct per-cell post. *)
#push-options ""
let rmsnorm_row_aux
  (bhw_n : nat) (c_n : pos)
  (eps inv_c : f32)
  (sx : chest2 f32 bhw_n c_n)
  (r : nat)
  : Lemma
          (requires
            r < bhw_n /\
            to_real eps >. 0.0R /\
            inv_c %~ (1.0R /. FStar.Real.of_int c_n))
          (ensures  row_rmsnormalized eps sx
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
    assert (acc1 sfac r == inv_rms_fn eps inv_c sumsq);
    let row = to_real_seq (EM.ematrix_row sx r) in
    let rss = frobenius_sumsq_r row in
    let rarg = rms_arg_r #c_n (to_real eps) row in
    a_mul sumsq inv_c rss (1.0R /. FStar.Real.of_int c_n);
    to_real_ok eps;
    a_add (mul sumsq inv_c) eps
      (rss *. (1.0R /. FStar.Real.of_int c_n)) (to_real eps);
    frobenius_sumsq_nonnegative row;
    assert (rss *. (1.0R /. FStar.Real.of_int c_n) ==
            rss /. FStar.Real.of_int c_n);
    assert (rarg >. 0.0R);
    RsqrtApprox.rsqrt_approx (add (mul sumsq inv_c) eps) rarg;
    let rinv = FStar.Math.Sqrt.rsqrt rarg in
    let out = Kuiper.Kernel.RowScale.s_row_scale sfac sx in
    let aux (j:nat{j<c_n}) : Lemma
      (acc2 out r j %~ ((row @! j) *. rinv)) =
      to_real_ok (acc2 sx r j);
      a_mul (acc2 sx r j) (acc1 sfac r) (row @! j) rinv
    in
    Classical.forall_intro aux
#pop-options

let rmsnorm_post_aux
  (bhw_n : nat) (c_n : pos)
  (eps inv_c : f32)
  (sx : chest2 f32 bhw_n c_n)
  : Lemma
      (requires
        to_real eps >. 0.0R /\
        inv_c %~ (1.0R /. FStar.Real.of_int c_n))
      (ensures rmsnorm_post bhw_n c_n eps sx
             (Kuiper.Kernel.RowScale.s_row_scale
                (chest_map (inv_rms_fn eps inv_c)
                   (seq_to_chest1 (HRed.seq_reduce_rows (sq_step #f32) sx)))
                sx))
  = let aux (r : nat { r < bhw_n }) : Lemma
      (row_rmsnormalized eps sx
        (Kuiper.Kernel.RowScale.s_row_scale
          (chest_map (inv_rms_fn eps inv_c)
            (seq_to_chest1 (HRed.seq_reduce_rows (sq_step #f32) sx)))
          sx)
        r)
      = rmsnorm_row_aux bhw_n c_n eps inv_c sx r
    in
    Classical.forall_intro aux

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn rmsnorm_fw_f32_impl
  (b : szp)
  (hw : SZ.t { 0 < SZ.v hw /\ SZ.fits (SZ.v b * SZ.v hw) })
  (c : SZ.t { 0 < SZ.v c /\ SZ.fits (SZ.v hw * SZ.v c) /\ SZ.fits (SZ.v b * (SZ.v hw * SZ.v c)) })
  (eps : f32)
  (inv_c : f32)
  (x : array2 f32 (l2_bcm_pages b hw c) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v hw) c)
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b * SZ.v hw > 0 /\
      SZ.v b * SZ.v hw * SZ.v c <= max_blocks * max_threads /\
      to_real eps >. 0.0R /\
      inv_c %~ (1.0R /. FStar.Real.of_int (SZ.v c))
    )
  ensures
    exists* (sx' : chest2 f32 (SZ.v b * SZ.v hw) c).
      on gpu_loc (x |-> sx') **
      pure (rmsnorm_post (SZ.v b * SZ.v hw) c eps sx sx')
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
  rmsnorm_post_aux bhw c eps inv_c sx;
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
  (x : array2 f32 (l2_bcm_pages b hw c) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v hw) c)
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b * SZ.v hw > 0 /\
      SZ.v b * SZ.v hw * SZ.v c <= max_blocks * max_threads /\
      to_real eps >. 0.0R
    )
  ensures
    exists* (sx' : chest2 f32 (SZ.v b * SZ.v hw) c).
      on gpu_loc (x |-> sx') **
      pure (rmsnorm_post (SZ.v b * SZ.v hw) c eps sx sx')
{
  let inv_c : f32 = rms_inv_c c;
  let c_i64 = FStar.Int.Cast.uint64_to_int64
    (FStar.SizeT.sizet_to_uint64 c);
  assert pure (FStar.Int64.v c_i64 == SZ.v c);
  of_int_approx #f32 c_i64;
  div_approx (one #f32) (of_int #f32 c_i64)
    1.0R (FStar.Real.of_int (SZ.v c));
  assert pure (inv_c %~ (1.0R /. FStar.Real.of_int (SZ.v c)));
  rmsnorm_fw_f32_impl b hw c eps inv_c x;
}

let rmsnorm_fw_f32 = rmsnorm_fw

fn rmsnorm4d_alloc_f32
  (b c h w : szp)
  (eps : f64)
  (x : array2 f32 (l2_bcm_pages b (h * w) c) { is_global x })
  (#f : perm)
  (#sx : chest2 f32 (SZ.v b * (SZ.v h * SZ.v w)) c)
  preserves cpu ** on gpu_loc (x |-> Frac f sx)
  requires
    pure (SZ.v h > 0 /\ SZ.v w > 0 /\ SZ.v c > 0 /\
          SZ.fits (SZ.v h * SZ.v w) /\
          SZ.fits (SZ.v b * (SZ.v h * SZ.v w)) /\
          SZ.fits ((SZ.v h * SZ.v w) * SZ.v c) /\
          SZ.fits (SZ.v b * ((SZ.v h * SZ.v w) * SZ.v c)) /\
          SZ.v b * (SZ.v h * SZ.v w) > 0 /\
          SZ.v b * (SZ.v h * SZ.v w) * SZ.v c <=
            max_blocks * max_threads /\
          to_real (fcast #f64 #f32 eps) >. 0.0R)
  returns out : array2 f32 (l2_bcm_pages b (h * w) c)
  ensures
    exists* (sx' : chest2 f32 (SZ.v b * (SZ.v h * SZ.v w)) c).
      on gpu_loc (out |-> sx') **
      pure (rmsnorm_post (SZ.v b * (SZ.v h * SZ.v w)) c
              (fcast #f64 #f32 eps) sx sx')
{
  let eps32 : f32 = fcast eps;
  let hw : szp = h *^ w;
  let bhw : szp = b *^ hw;
  let elems : szp = bhw *^ c;
  let out = Copy.copy_alloc #f32 elems x;
  rmsnorm_fw_f32 b hw c eps32 out;
  out
}
