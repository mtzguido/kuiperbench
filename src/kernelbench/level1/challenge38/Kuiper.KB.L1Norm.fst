module Kuiper.KB.L1Norm

(* KernelBench L1 #38: L1 normalisation, 3 GPU launches.

   Launch 1: reduce_batched fabs     → sum_abs[r] = Σ_j |x[r,j]|
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
module Map = Kuiper.Kernel.Map
module HRed = Kuiper.Kernel.HReduce
module RowScale = Kuiper.Kernel.RowScale
module KS = Kuiper.Seq.Common

(* post_map for the per-row scale factor D / s.  The pre_map for the
   reduction itself is [fabs], reused so that the bridge lemma
   [row_reduce_partial_fabs_approx] applies unchanged. *)
inline_for_extraction noextract
let l1_scale_fn (dim_f : f32) (s : f32) : f32 = div dim_f s

(* Per-row functional postcondition.  After the three GPU launches the
   array [x] holds, by composition of the three primitives' specs, the
   matrix
     s_row_scale (lseq_map (l1_scale_fn dim_f)
                            (seq_reduce_rows fabs sx))
                 sx
   The witness for [row_l1_normalized] at row [r] is then
     sum_abs_r = row_reduce_partial fabs sx r d
     scale_r   = div dim_f sum_abs_r
   The [sum_abs_r %~ l1_sum_r ...] obligation reduces to
   [row_reduce_partial_fabs_approx]. *)
#push-options ""
let l1norm_row_aux
  (b_n d_n : nat)
  (dim_f : f32)
  (sx : EM.chest2 f32 b_n d_n)
  (r : nat)
  : Lemma (requires r < b_n)
          (ensures  row_l1_normalized dim_f sx
                      (Kuiper.Kernel.RowScale.s_row_scale
                         (chest_map (l1_scale_fn dim_f)
                            (seq_to_chest1 (HRed.seq_reduce_rows (fabs #f32) sx)))
                         sx)
                      r)
  = row_reduce_partial_fabs_approx sx r;
    let sfac : chest1 f32 b_n =
      chest_map (l1_scale_fn dim_f)
                (seq_to_chest1 (HRed.seq_reduce_rows (fabs #f32) sx)) in
    let sum_abs : f32 = HRed.row_reduce_partial (fabs #f32) sx r d_n in
    (* acc1 sfac r reduces through chest_map / seq_to_chest1 / init_ghost
       to [div dim_f (row_reduce_partial fabs sx r d_n)]. *)
    assert (acc1 sfac r == div dim_f sum_abs)
#pop-options

let l1norm_post_aux
  (b_n d_n : nat)
  (dim_f : f32)
  (sx : EM.chest2 f32 b_n d_n)
  : Lemma (l1norm_post b_n d_n dim_f sx
             (Kuiper.Kernel.RowScale.s_row_scale
                (chest_map (l1_scale_fn dim_f)
                   (seq_to_chest1 (HRed.seq_reduce_rows (fabs #f32) sx)))
                sx))
  = let aux (r : nat { r < b_n }) : Lemma
      (row_l1_normalized dim_f sx
        (Kuiper.Kernel.RowScale.s_row_scale
          (chest_map (l1_scale_fn dim_f)
            (seq_to_chest1 (HRed.seq_reduce_rows (fabs #f32) sx)))
          sx)
        r)
      = l1norm_row_aux b_n d_n dim_f sx r
    in
    Classical.forall_intro aux

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn l1norm_fw_f32_impl
  (b : szp)
  (d : szp { 0 < SZ.v d /\ SZ.fits (SZ.v b * SZ.v d) })
  (dim_f : f32)
  (x : array2 f32 (l2_row_major (SZ.v b) (SZ.v d)) { is_global x })
  (#sx : EM.chest2 f32 (SZ.v b) (SZ.v d))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b > 0 /\
      SZ.v b * SZ.v d <= max_blocks * max_threads
    )
  ensures
    exists* (sx' : EM.chest2 f32 (SZ.v b) (SZ.v d)).
      on gpu_loc (x |-> sx') **
      pure (l1norm_post (SZ.v b) (SZ.v d) dim_f sx sx')
{
  (* b <= b * d  since d >= 1 *)
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v d) 1 (SZ.v b);

  (* ── Launch 1: sum of absolutes per row ── *)
  let sum_abs = alloc0 #f32 b (l1_forward b);
  with em. assert (on gpu_loc (sum_abs |-> em));

  assert pure (SZ.v b <= max_blocks * max_threads);
  HRed.reduce_batched (fabs #f32) b d #_ #(c_l2_row_major (SZ.v b) d) x sum_abs;

  (* ── Launch 2: per-row scale factor D / s ── *)
  Map.map_gpu (l1_scale_fn dim_f) b sum_abs;

  (* ── Launch 3: in-place row scale ── *)
  assert pure (SZ.v b * SZ.v d <= max_blocks * max_threads);
  RowScale.row_scale b d sum_abs #_ #(c_l2_row_major (SZ.v b) d) x;

  (* Free scratch buffer; row_scale preserves sum_abs at full perm *)
  free sum_abs;

  (* Discharge the per-row functional postcondition. *)
  l1norm_post_aux (SZ.v b) (SZ.v d) dim_f sx;
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
  (x : array2 f32 (l2_row_major (SZ.v b) (SZ.v d)) { is_global x })
  (#sx : EM.chest2 f32 (SZ.v b) (SZ.v d))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    pure (
      SZ.v b > 0 /\
      SZ.v b * SZ.v d <= max_blocks * max_threads
    )
  ensures
    exists* (sx' : EM.chest2 f32 (SZ.v b) (SZ.v d)).
      on gpu_loc (x |-> sx') **
      pure (l1norm_post (SZ.v b) (SZ.v d) (l1_dim_f d) sx sx')
{
  let dim_f : f32 = l1_dim_f d;
  l1norm_fw_f32_impl b d dim_f x;
}

let l1norm_fw_f32 : l1norm_fw_ty f32 = l1norm_fw
