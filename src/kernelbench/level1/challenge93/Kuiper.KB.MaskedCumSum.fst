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

ghost
fn reshape2to1
  (#et:Type) (#m #cn:nat)
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 et (l2_row_major m cn))
  (#s2 : chest2 et m cn)
  (#f : perm)
  requires a2 |-> Frac f s2
  ensures
    from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l2_row_major m cn) s2))
{
  tensor_concr a2;
  tensor_abs' (l1_forward p) (core a2)
}

ghost
fn reshape1to2
  (#et:Type) (#m #cn:nat)
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 et (l2_row_major m cn))
  (#s2 : chest2 et m cn)
  (#f : perm)
  requires
    from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l2_row_major m cn) s2))
  ensures a2 |-> Frac f s2
{
  tensor_concr (from_array (l1_forward p) (core a2));
  rewrite
    (core (from_array (l1_forward p) (core a2))
      |-> Frac f (to_seq (l1_forward p)
                    (from_seq (l1_forward p)
                       (to_seq (l2_row_major m cn) s2))))
  as (core a2 |-> Frac f (to_seq (l2_row_major m cn) s2));
  tensor_abs (l2_row_major m cn) (core a2) #f #s2;
  rewrite (from_array (l2_row_major m cn) (core a2) |-> Frac f s2)
       as (a2 |-> Frac f s2)
}

let flat_mask_map
  (#b #d #n : nat)
  (_ : squash (n == b * d))
  (sx : chest2 f32 b d)
  (sm : chest2 u8 b d)
  : Lemma
      (mk1 (fun i ->
         mask_step_f32
           (acc1 (from_seq (l1_forward n)
                    (to_seq (l2_row_major b d) sx)) i)
           (acc1 (from_seq (l1_forward n)
                    (to_seq (l2_row_major b d) sm)) i))
       == from_seq (l1_forward n)
            (to_seq (l2_row_major b d) (masked_chest sx sm)))
  = let lhs = mk1 (fun i ->
      mask_step_f32
        (acc1 (from_seq (l1_forward n)
                 (to_seq (l2_row_major b d) sx)) i)
        (acc1 (from_seq (l1_forward n)
                 (to_seq (l2_row_major b d) sm)) i)) in
    let rhs = from_seq (l1_forward n)
      (to_seq (l2_row_major b d) (masked_chest sx sm)) in
    let aux (i : natlt n) : Lemma (acc1 lhs i == acc1 rhs i) = () in
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

  map_loc gpu_loc (fun () -> reshape2to1 (SZ.v n) input);
  map_loc gpu_loc (fun () -> reshape2to1 (SZ.v n) mask);
  map_loc gpu_loc (fun () -> reshape2to1 (SZ.v n) scratch);
  Map.map_gpu2_to #f32 #u8 #f32 mask_step_f32 n
    #(l1_forward (SZ.v n)) #(c_l1_forward (FStar.Ghost.hide (SZ.v n)))
    #(l1_forward (SZ.v n)) #(c_l1_forward (FStar.Ghost.hide (SZ.v n)))
    #(l1_forward (SZ.v n)) #(c_l1_forward (FStar.Ghost.hide (SZ.v n)))
    (from_array (l1_forward (SZ.v n)) (core input))
    (from_array (l1_forward (SZ.v n)) (core mask))
    (from_array (l1_forward (SZ.v n)) (core scratch));

  flat_mask_map #(SZ.v b) #(SZ.v d) #(SZ.v n) ()
    (reveal sx) (reveal sm);
  map_loc gpu_loc (fun () -> reshape1to2 (SZ.v n) input);
  map_loc gpu_loc (fun () -> reshape1to2 (SZ.v n) mask);
  map_loc gpu_loc (fun () -> reshape1to2 (SZ.v n) scratch
    #(masked_chest (reveal sx) (reveal sm)) #_);

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
