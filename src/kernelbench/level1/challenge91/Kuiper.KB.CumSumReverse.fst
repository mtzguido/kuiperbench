module Kuiper.KB.CumSumReverse

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Bijection
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

(* Involutive reversal of the inner coordinate of a [(b,d)] index. *)
let reverse_inner_bij (b d : nat)
  : (abs (b @| d @| INil) =~ abs (b @| d @| INil))
  = {
      ff = (fun (x : abs (b @| d @| INil)) ->
        let r, (i, ()) = x in
        (r, ((d - 1 - i <: natlt d), ())));
      gg = (fun (x : abs (b @| d @| INil)) ->
        let r, (i, ()) = x in
        (r, ((d - 1 - i <: natlt d), ())));
      ff_gg = (fun (x : abs (b @| d @| INil)) ->
        let r, (i, ()) = x in ());
      gg_ff = (fun (x : abs (b @| d @| INil)) ->
        let r, (i, ()) = x in ());
    }

(* Compute the reversed row-major physical offset directly.  Going through
   [ctlayout_bij] would first construct a reversed coordinate pair and then
   immediately flatten it, leaving an anonymous tuple in extracted CUDA. *)
inline_for_extraction noextract
let reverse_inner_cimap
  (b : szp) (d : szp { SZ.fits (SZ.v b * SZ.v d) })
  (x : conc (b @| d @| INil))
  : r:SZ.t {
      SZ.v r ==
        (tlayout_bij (reverse_inner_bij (SZ.v b) (SZ.v d))
          (l2_row_major b d)).imap.f (up x) }
  = let row, (col, ()) = x in
    SZ.(row *^ d +^ (d -^ 1sz -^ col))

inline_for_extraction noextract
instance c_reverse_inner_layout
  (#b : szp) (#d : szp { SZ.fits (SZ.v b * SZ.v d) })
  : ctlayout
      (tlayout_bij (reverse_inner_bij (SZ.v b) (SZ.v d))
        (l2_row_major b d))
  = {
      ulen_fits = ();
      all_fit = ();
      cimap = reverse_inner_cimap b d;
    }

let reverse_rows_chest (#t : Type0) (#b #d : nat)
  (s : chest2 t b d)
  : chest2 t b d
  = mk (b @| d @| INil)
      (fun i -> acc s (i <~| reverse_inner_bij b d))

let reverse_rows_row (#t : Type0) (#b #d : nat)
  (s : chest2 t b d) (r : natlt b)
  : Lemma
      (EM.ematrix_row (reverse_rows_chest s) r `Seq.equal`
       seq_rev (EM.ematrix_row s r))
  = ()

let seq_rev_to_real
  (#t : Type0) {| scalar t, real_like t |}
  (s : Seq.seq t)
  : Lemma
      (to_real_seq (seq_rev s) `Seq.equal`
       seq_rev (to_real_seq s))
  = ()

let slice_to_real_seq
  (#t : Type0) {| scalar t, real_like t |}
  (s : Seq.seq t)
  (k : nat { k <= Seq.length s })
  : Lemma
      (to_real_seq (Seq.slice s 0 k) `Seq.equal`
       Seq.slice (to_real_seq s) 0 k)
  = ()

let slice_is_approx
  (#t : Type0) {| scalar t, real_like t |}
  (s : Seq.seq t)
  (k : nat { k <= Seq.length s })
  : Lemma
      (Seq.slice s 0 k %~ Seq.slice (to_real_seq s) 0 k)
  = let lhs : Seq.seq t = Seq.slice s 0 k in
    to_real_seq_is_approx lhs;
    slice_to_real_seq #t s k;
    Seq.lemma_eq_elim (to_real_seq lhs)
      (Seq.slice (to_real_seq s) 0 k)

#push-options "--z3rlimit 60"
let reverse_cell_post_eq
  (#b #d : nat)
  (sx : chest2 f32 b d)
  (r : natlt b)
  (i : natlt d)
  : Lemma
      (acc2
         (reverse_rows_chest
           (scan2d_inclusive_result reducer_fadd_f32
             (reverse_rows_chest sx)))
         r i
       %~
       rsum
         (Seq.slice
           (seq_rev (to_real_seq (EM.ematrix_row sx r)))
           0 (d - i)))
  = let revrow = EM.ematrix_row (reverse_rows_chest sx) r in
    let k = d - i in
    reverse_rows_row sx r;
    seq_rev_to_real (EM.ematrix_row sx r);
    slice_is_approx #f32 revrow k;
    sum_is_approx #f32
      (Seq.slice revrow 0 k)
      (Seq.slice (to_real_seq revrow) 0 k);
    reducer_fadd_f32_proj ();
    assert
      (acc2
         (reverse_rows_chest
           (scan2d_inclusive_result reducer_fadd_f32
             (reverse_rows_chest sx)))
         r i
       == SC.seq_fold_left (add #f32) (zero #f32)
            (Seq.slice revrow 0 k));
    assert
      (SC.seq_fold_left (add #f32) (zero #f32)
         (Seq.slice revrow 0 k)
       %~ rsum (Seq.slice (to_real_seq revrow) 0 k));
    Seq.lemma_eq_elim (to_real_seq revrow)
      (seq_rev (to_real_seq (EM.ematrix_row sx r)))
#pop-options

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn cumsum_reverse_fw_f32_impl
  (b : szp { b <= max_blocks })
  (d : szp { SZ.fits (SZ.v b * SZ.v d) })
  (input  : array2 f32 (l2_row_major b d) { is_global input })
  (output : array2 f32 (l2_row_major b d) { is_global output })
  (#sx #sy0 : chest2 f32 b d)
  preserves
    cpu **
    on gpu_loc (input |-> sx)
  requires
    on gpu_loc (output |-> sy0)
  ensures
    exists* (sy : chest2 f32 b d).
      on gpu_loc (output |-> sy) **
      pure (cumsum_reverse_post b d sx sy)
{
  let input_r = tensor_apply_bij_ro_located
    (reverse_inner_bij (SZ.v b) (SZ.v d)) input;
  let output_r = tensor_apply_bij_st_located
    (reverse_inner_bij (SZ.v b) (SZ.v d)) output;

  scan1d_inclusive_rowblock #f32 reducer_fadd_f32 b d
    #(tlayout_bij (reverse_inner_bij (SZ.v b) (SZ.v d))
        (l2_row_major b d))
    #(c_reverse_inner_layout #b #d)
    #(tlayout_bij (reverse_inner_bij (SZ.v b) (SZ.v d))
        (l2_row_major b d))
    #(c_reverse_inner_layout #b #d)
    input_r output_r;

  Pulse.Lib.Forall.elim_forall
    (scan2d_inclusive_result reducer_fadd_f32
      (reverse_rows_chest (reveal sx)) <:
      chest2 f32 b d);
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc
      (output_r |->
        scan2d_inclusive_result reducer_fadd_f32
          (reverse_rows_chest (reveal sx))))
    _;
  Pulse.Lib.Trade.elim_trade
    (on gpu_loc (input_r |-> reverse_rows_chest (reveal sx))) _;

  Classical.forall_intro_2
    (Classical.move_requires_2
      (reverse_cell_post_eq #b #d (reveal sx)));
  ()
}
#pop-options

#push-options "--z3rlimit 60"
fn cumsum_reverse_fw_f32
  (b : szp { b <= max_blocks })
  (d : szp { SZ.fits (SZ.v b * SZ.v d) })
  (input : array2 f32 (l2_row_major b d) { is_global input })
  (#sx : chest2 f32 b d)
  preserves
    cpu **
    on gpu_loc (input |-> sx)
  returns output : array2 f32 (l2_row_major b d)
  ensures
    exists* (sy : chest2 f32 b d).
      on gpu_loc (output |-> sy) **
      pure (cumsum_reverse_post b d sx sy)
{
  let n : szp = b *^ d;
  let output = alloc0 #f32 n (l2_row_major b d);
  with sy0. assert on gpu_loc (output |-> sy0);
  cumsum_reverse_fw_f32_impl b d input output;
  output
}
#pop-options
