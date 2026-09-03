module Kuiper.KB.L1Norm

(* KernelBench L1 #38: L1 normalisation, 3 GPU launches.

   Launch 1: reduce_batched l1_abs   → sum_abs[r] = Σ_j |x[r,j]|
   Launch 2: map_gpu l1_scale_fn      → sum_abs[r] = D / sum_abs[r]
   Launch 3: row_scale sum_abs x      → x[r,j] ← x[r,j] * sum_abs[r]

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, c_l2_row_major }
open Kuiper.Spec.L1Norm
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Copy = Kuiper.KB.Tensor.Copy
module Map = Kuiper.Kernel.Map
module HRed = Kuiper.Kernel.HReduce
module RowScale = Kuiper.Kernel.RowScale
module KS = Kuiper.Seq.Common

(* post_map for the per-row scale factor D / s.  The reduction pre-map is
   [l1_abs], whose real approximation is proved in Kuiper.Spec.L1Norm. *)
inline_for_extraction noextract
let l1_scale_fn (dim_f : f32) (s : f32) : f32 = div dim_f s

(* Per-row functional postcondition.  After the three GPU launches the
   array [x] holds, by composition of the three primitives' specs, the
   matrix
     s_row_scale (lseq_map (l1_scale_fn dim_f)
                            (seq_reduce_rows l1_abs sx))
                 sx
   The proof relates the reduction and division intermediates to the
   direct real per-cell postcondition over [rx]. *)
#push-options ""
let l1norm_row_aux
  (b_n : nat) (d_n : pos)
  (dim_f : f32)
  (sx : chest2 f32 b_n d_n)
  (rx : chest2 real b_n d_n)
  (r : nat)
  : Lemma
          (requires
            r < b_n /\ sx %~ rx /\ l1norm_domain rx /\
            dim_f %~ FStar.Real.of_int d_n)
          (ensures  row_l1_normalized rx
                      (Kuiper.Kernel.RowScale.s_row_scale
                         (chest_map (l1_scale_fn dim_f)
                            (seq_to_chest1 (HRed.seq_reduce_rows (l1_abs #f32) sx)))
                         sx)
                      r)
  = row_reduce_partial_l1_abs_approx sx rx r;
    let sfac : chest1 f32 b_n =
      chest_map (l1_scale_fn dim_f)
                (seq_to_chest1 (HRed.seq_reduce_rows (l1_abs #f32) sx)) in
    let sum_abs : f32 = HRed.row_reduce_partial (l1_abs #f32) sx r d_n in
    (* acc1 sfac r reduces through chest_map / seq_to_chest1 / init_ghost
       to [div dim_f (row_reduce_partial l1_abs sx r d_n)]. *)
    assert (acc1 sfac r == div dim_f sum_abs);
    let row = EM.ematrix_row rx r in
    let rsum = l1_sum_r row in
    div_approx dim_f sum_abs (FStar.Real.of_int d_n) rsum;
    let rscale = l1_scale_r #d_n row in
    let out = Kuiper.Kernel.RowScale.s_row_scale sfac sx in
    let aux (j:nat{j<d_n}) : Lemma
      (acc2 out r j %~ (acc2 rx r j *. rscale)) =
      assert (acc2 sx r j %~ acc2 rx r j);
      a_mul (acc2 sx r j) (acc1 sfac r)
        (acc2 rx r j) rscale
    in
    Classical.forall_intro aux
#pop-options

let l1norm_post_aux
  (b_n : nat) (d_n : pos)
  (dim_f : f32)
  (sx : chest2 f32 b_n d_n)
  (rx : chest2 real b_n d_n)
  : Lemma
      (requires sx %~ rx /\ l1norm_domain rx /\
                dim_f %~ FStar.Real.of_int d_n)
      (ensures l1norm_post b_n d_n rx
             (Kuiper.Kernel.RowScale.s_row_scale
                (chest_map (l1_scale_fn dim_f)
                   (seq_to_chest1 (HRed.seq_reduce_rows (l1_abs #f32) sx)))
                sx))
  = let aux (r : nat { r < b_n }) : Lemma
      (row_l1_normalized rx
        (Kuiper.Kernel.RowScale.s_row_scale
          (chest_map (l1_scale_fn dim_f)
            (seq_to_chest1 (HRed.seq_reduce_rows (l1_abs #f32) sx)))
          sx)
        r)
      = l1norm_row_aux b_n d_n dim_f sx rx r
    in
    Classical.forall_intro aux

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn l1norm_fw_f32_impl
  (b : szp)
  (d : szp { 0 < SZ.v d /\ SZ.fits (SZ.v b * SZ.v d) })
  (dim_f : f32)
  (x : array2 f32 (l2_row_major b d) { is_global x })
  (#sx : chest2 f32 b d)
  (rx : erased (chest2 real b d))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b > 0 /\
      SZ.v b * SZ.v d <= max_blocks * max_threads /\
      sx %~ reveal rx /\
      l1norm_domain (reveal rx) /\
      dim_f %~ FStar.Real.of_int (SZ.v d)
    )
  ensures
    exists* (sx' : chest2 f32 b d).
      on gpu_loc (x |-> sx') **
      pure (l1norm_post b d (reveal rx) sx')
{
  (* b <= b * d  since d >= 1 *)
  FStar.Math.Lemmas.lemma_mult_le_right d 1 b;

  (* ── Launch 1: sum of absolutes per row ── *)
  let sum_abs = alloc0 #f32 b (l1_forward b);
  with em. assert (on gpu_loc (sum_abs |-> em));

  assert pure (SZ.v b <= max_blocks * max_threads);
  HRed.reduce_batched (l1_abs #f32) b d #_ #(c_l2_row_major (SZ.v b) d) x sum_abs;

  (* ── Launch 2: per-row scale factor D / s ── *)
  Map.map_gpu (l1_scale_fn dim_f) b sum_abs;

  (* ── Launch 3: in-place row scale ── *)
  assert pure (SZ.v b * SZ.v d <= max_blocks * max_threads);
  RowScale.row_scale b d sum_abs #_ #(c_l2_row_major (SZ.v b) d) x;

  (* Free scratch buffer; row_scale preserves sum_abs at full perm *)
  free sum_abs;

  (* Discharge the per-row functional postcondition. *)
  l1norm_post_aux b d dim_f sx (reveal rx);
  ()
}
#pop-options

(* Public entry point: compute the float-valued dimension [l1_dim_f d]
   inside the verification boundary (extracts to
   (float)(int64_t)(uint64_t)D), then delegate to [l1norm_fw_f32_impl].
   The proof above treats [dim_f] abstractly, so this constant
   computation does not affect its cost. *)
fn l1norm_fw
  (b : szp)
  (d : szp { 0 < SZ.v d /\ SZ.fits (SZ.v b * SZ.v d) })
  (x : array2 f32 (l2_row_major b d) { is_global x })
  (#sx : chest2 f32 b d)
  (rx : erased (chest2 real b d))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b > 0 /\
      SZ.v b * SZ.v d <= max_blocks * max_threads /\
      sx %~ reveal rx /\
      l1norm_domain (reveal rx)
    )
  ensures
    exists* (sx' : chest2 f32 b d).
      on gpu_loc (x |-> sx') **
      pure (l1norm_post b d (reveal rx) sx')
{
  let dim_f : f32 = l1_dim_f d;
  let d_i64 = FStar.Int.Cast.uint64_to_int64
    (FStar.SizeT.sizet_to_uint64 d);
  assert pure (FStar.Int64.v d_i64 == SZ.v d);
  of_int_approx #f32 d_i64;
  assert pure (dim_f %~ FStar.Real.of_int (SZ.v d));
  l1norm_fw_f32_impl b d dim_f x rx;
}

let l1norm_fw_f32 = l1norm_fw

fn l1norm_alloc_f32
  (b : szp)
  (d : szp { 0 < SZ.v d /\ SZ.fits (SZ.v b * SZ.v d) })
  (x : array2 f32 (l2_row_major b d) { is_global x })
  (#f : perm)
  (#sx : chest2 f32 b d)
  (rx : erased (chest2 real b d))
  preserves cpu ** on gpu_loc (x |-> Frac f sx)
  requires
    pure (SZ.v b > 0 /\
          SZ.v b * SZ.v d <= max_blocks * max_threads /\
          sx %~ reveal rx /\
          l1norm_domain (reveal rx))
  returns out : array2 f32 (l2_row_major b d)
  ensures
    exists* (sx' : chest2 f32 b d).
      on gpu_loc (out |-> sx') **
      pure (l1norm_post b d (reveal rx) sx')
{
  let n : szp = b *^ d;
  let out = Copy.copy_alloc #f32 n x;
  l1norm_fw_f32 b d out rx;
  out
}
