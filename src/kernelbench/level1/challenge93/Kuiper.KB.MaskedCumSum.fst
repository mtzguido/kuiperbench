module Kuiper.KB.MaskedCumSum

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.Bijection
open Kuiper.Spec.Scan1D
open Kuiper.Monoid.Reduce.F32
open Kuiper.Kernel.Scan1D.RowBlock
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq
module SC = Kuiper.Seq.Common
module Map = Kuiper.Kernel.Map

inline_for_extraction noextract
let mask_step_f32 (x : f32) (m : u8) : f32 =
  if m = 0uy then zero else x

let masked_chest
  (#b #d : nat)
  (sx : chest2 f32 b d)
  (sm : chest2 u8 b d)
  : chest2 f32 b d
  = mk2 (fun r i -> mask_step_f32 (acc2 sx r i) (acc2 sm r i))

let mask_step_approx (x : f32) (m : u8)
  : Lemma
      (mask_step_f32 x m %~ real_mask_step (to_real x) m)
  = if m = 0uy
    then begin
      to_real_ok (zero #f32);
      assert ((zero #f32) %~ 0.0R)
    end
    else to_real_ok x

let masked_row_approx
  (#b #d : nat)
  (sx : chest2 f32 b d)
  (sm : chest2 u8 b d)
  (r : natlt b)
  : Lemma
      (EM.ematrix_row (masked_chest sx sm) r %~
       real_masked_row sx sm r)
  = let aux (i : natlt d)
      : Lemma
          (Seq.index (EM.ematrix_row (masked_chest sx sm) r) i %~
           Seq.index (real_masked_row sx sm r) i)
      = mask_step_approx (acc2 sx r i) (acc2 sm r i)
    in
    Classical.forall_intro aux

let slice_approx
  (#t : Type0) {| scalar t, real_like t |}
  (s : Seq.seq t)
  (rs : Seq.seq real)
  (k : nat { k <= Seq.length s })
  : Lemma
      (requires s %~ rs /\ Seq.length rs == Seq.length s)
      (ensures Seq.slice s 0 k %~ Seq.slice rs 0 k)
  = ()

let macc_scan2d_inclusive_result_f32
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  (r : natlt rows) (i : natlt cols)
  : Lemma
      (acc2 (scan2d_inclusive_result reducer_fadd_f32 sx) r i
       == scan_inclusive_at reducer_fadd_f32 (EM.ematrix_row sx r) i)
  = ()

#push-options "--z3rlimit 60"
let masked_cell_post_eq
  (#b #d : nat)
  (sx : chest2 f32 b d)
  (sm : chest2 u8 b d)
  (r : natlt b)
  (i : natlt d)
  : Lemma
      (acc2
         (scan2d_inclusive_result reducer_fadd_f32
           (masked_chest sx sm))
         r i
       %~ rsum (Seq.slice (real_masked_row sx sm r) 0 (i + 1)))
  = let row = EM.ematrix_row (masked_chest sx sm) r in
    let rrow = real_masked_row sx sm r in
    masked_row_approx sx sm r;
    slice_approx #f32 row rrow (i + 1);
    sum_is_approx #f32
      (Seq.slice row 0 (i + 1))
      (Seq.slice rrow 0 (i + 1));
    macc_scan2d_inclusive_result_f32 (masked_chest sx sm) r i;
    reducer_fadd_f32_proj ();
    assert
      (acc2
         (scan2d_inclusive_result reducer_fadd_f32
           (masked_chest sx sm))
         r i
       == SC.seq_fold_left (add #f32) (zero #f32)
            (Seq.slice row 0 (i + 1)))
#pop-options

let unfolded_mask_map
  (#b #d : nat)
  (sx : chest2 f32 b d)
  (sm : chest2 u8 b d)
  : Lemma
      (unfold_chest
         (mk1 (fun i ->
           mask_step_f32
             (acc1 (fold_chest sx) i)
             (acc1 (fold_chest sm) i)))
       == masked_chest sx sm)
  = let lhs = unfold_chest
      (mk1 (fun i ->
        mask_step_f32
          (acc1 (fold_chest sx) i)
          (acc1 (fold_chest sm) i))) in
    let rhs = masked_chest sx sm in
    let aux (i : abs (b @| d @| INil))
      : Lemma (acc lhs i == acc rhs i)
      = ()
    in
    Classical.forall_intro aux;
    Kuiper.Chest.lemma_equal_intro lhs rhs;
    Kuiper.Chest.ext lhs rhs

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn masked_cumsum_fw_f32_impl
  (b : szp { b <= max_blocks })
  (d : szp {
    SZ.fits (SZ.v b * SZ.v d) /\
    SZ.v b * SZ.v d <= max_blocks * max_threads })
  (input : array2 f32 (l2_row_major b d) { is_global input })
  (mask  : array2 u8  (l2_row_major b d) { is_global mask })
  (output : array2 f32 (l2_row_major b d) { is_global output })
  (#sx : chest2 f32 b d)
  (#sm : chest2 u8 b d)
  (#sy0 : chest2 f32 b d)
  preserves
    cpu **
    on gpu_loc (input |-> sx) **
    on gpu_loc (mask |-> sm)
  requires
    on gpu_loc (output |-> sy0)
  ensures
    exists* (sy : chest2 f32 b d).
      on gpu_loc (output |-> sy) **
      pure (masked_cumsum_post b d sx sm sy)
{
  let n : szp = b *^ d;
  let scratch = alloc0 #f32 n (l2_row_major b d);

  let input_f = tensor_fold_ro_located input;
  let mask_f = tensor_fold_ro_located mask;
  let scratch_f = tensor_fold_st_located scratch;

  Map.map_gpu2_to #f32 #u8 #f32 mask_step_f32 n
    #(tlayout_fold_outer (l2_row_major b d))
    #(ctlayout_fold_outer (l2_row_major b d))
    #(tlayout_fold_outer (l2_row_major b d))
    #(ctlayout_fold_outer (l2_row_major b d))
    #(tlayout_fold_outer (l2_row_major b d))
    #(ctlayout_fold_outer (l2_row_major b d))
    input_f mask_f scratch_f;

  with mapped_f. assert on gpu_loc (scratch_f |-> mapped_f);
  Pulse.Lib.Forall.elim_forall
    (mapped_f <: chest (fold_outer (b @| d @| INil)) f32);
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc
      (scratch_f |->
        (mapped_f <: chest (fold_outer (b @| d @| INil)) f32))) _;
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc (mask_f |-> fold_chest (reveal sm))) _;
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc (input_f |-> fold_chest (reveal sx))) _;

  unfolded_mask_map (reveal sx) (reveal sm);
  rewrite each
    (unfold_chest #f32 #2 #(b @| d @| INil)
      (mapped_f <: chest (fold_outer (b @| d @| INil)) f32))
    as masked_chest (reveal sx) (reveal sm);

  scan1d_inclusive_rowblock #f32 reducer_fadd_f32 b d
    #(l2_row_major b d) #_
    #(l2_row_major b d) #_
    scratch output;

  free scratch;

  Classical.forall_intro_2
    (Classical.move_requires_2
      (masked_cell_post_eq #b #d (reveal sx) (reveal sm)));
  ()
}
#pop-options

#push-options "--z3rlimit 60"
fn masked_cumsum_fw_f32
  (b : szp { b <= max_blocks })
  (d : szp {
    SZ.fits (SZ.v b * SZ.v d) /\
    SZ.v b * SZ.v d <= max_blocks * max_threads })
  (input : array2 f32 (l2_row_major b d) { is_global input })
  (mask  : array2 u8  (l2_row_major b d) { is_global mask })
  (#sx : chest2 f32 b d)
  (#sm : chest2 u8 b d)
  preserves
    cpu **
    on gpu_loc (input |-> sx) **
    on gpu_loc (mask |-> sm)
  returns output : array2 f32 (l2_row_major b d)
  ensures
    exists* (sy : chest2 f32 b d).
      on gpu_loc (output |-> sy) **
      pure (masked_cumsum_post b d sx sm sy)
{
  let n : szp = b *^ d;
  let output = alloc0 #f32 n (l2_row_major b d);
  with sy0. assert on gpu_loc (output |-> sy0);
  masked_cumsum_fw_f32_impl b d input mask output;
  output
}
#pop-options
