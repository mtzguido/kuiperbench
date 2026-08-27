module Kuiper.KB.CumSumExclusive

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Spec.Scan1D
open Kuiper.Monoid.Reduce
open Kuiper.Monoid.Reduce.F32
open Kuiper.Kernel.Scan1D.RowBlockExcl
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq
module SC = Kuiper.Seq.Common

(* Slice/to_real_seq commute: [to_real_seq (slice s 0 k) == slice (to_real_seq s) 0 k]
   as length-[k] sequences. *)
let slice_to_real_seq
  (#t:Type0) {| scalar t, real_like t |}
  (s : Seq.seq t)
  (k : nat { k <= Seq.length s })
  : Lemma (to_real_seq (Seq.slice s 0 k) `Seq.equal`
           Seq.slice (to_real_seq s) 0 k)
  = ()

(* Bridge: a length-[k] slice of [row sx r] is approximated by the same
   slice of [to_real_seq (row sx r)]. *)
let slice_is_approx
  (#t:Type0) {| scalar t, real_like t |}
  (s : Seq.seq t)
  (k : nat { k <= Seq.length s })
  : Lemma (Seq.slice s 0 k %~ Seq.slice (to_real_seq s) 0 k)
  = let lhs : Seq.seq t = Seq.slice s 0 k in
    to_real_seq_is_approx lhs;
    slice_to_real_seq #t s k;
    Seq.lemma_eq_elim (to_real_seq lhs) (Seq.slice (to_real_seq s) 0 k)

(* Mirror of [Kuiper.Kernel.Scan1D.RowBlockExcl.macc_scan2d_exclusive_result],
   not exported via that module's .fsti. *)
let macc_scan2d_exclusive_result_f32
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows) (i : natlt cols)
  : Lemma (acc2 (scan2d_exclusive_result cmonoid_fadd_f32 sx) r i
           == scan_exclusive_at cmonoid_fadd_f32 (EM.ematrix_row sx r) i)
  = ()

(* z3rlimit > 40: the per-cell %~ bridge chains slice/sum approximation
 * lemmas; split_queries + 60 mirrors the inclusive [cell_post_eq] in
 * Kuiper.KB.CumSum. *)
#push-options " --z3rlimit 60"
let cell_post_eq
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (i : natlt cols)
  : Lemma
      (acc2 (scan2d_exclusive_result cmonoid_fadd_f32 sx) r i
       %~ rsum (Seq.slice (to_real_seq (EM.ematrix_row sx r)) 0 i))
  = let row : Seq.seq f32 = EM.ematrix_row sx r in
    let s  : Seq.seq f32 = Seq.slice row 0 i in
    let s' : Seq.seq real = Seq.slice (to_real_seq row) 0 i in
    slice_is_approx #f32 row i;
    sum_is_approx #f32 s s';
    macc_scan2d_exclusive_result_f32 sx r i;
    cmonoid_fadd_f32_proj ();
    assert (acc2 (scan2d_exclusive_result cmonoid_fadd_f32 sx) r i
              == scan_exclusive_at cmonoid_fadd_f32 row i);
    assert (scan_exclusive_at cmonoid_fadd_f32 row i
              == SC.seq_fold_left add zero s);
    assert (SC.seq_fold_left add zero s %~ rsum s')
#pop-options

(* z3rlimit > 40: single kernel launch + Classical.forall_intro_2 of the
 * per-cell bridge; mirrors the inclusive [cumsum_fw_f32_impl]. *)
#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn cumsum_exclusive_fw_f32_impl
  (b : szp { b <= max_blocks })
  (d : szp { SZ.fits (SZ.v b * SZ.v d) })
  (input  : array2 f32 (l2_row_major b d)
            { is_global input  })
  (output : array2 f32 (l2_row_major b d)
            { is_global output })
  (#sx  : EM.chest2 f32 b d)
  (#sy0 : EM.chest2 f32 b d)
  requires
    cpu **
    on gpu_loc (input  |-> sx) **
    on gpu_loc (output |-> sy0)
  ensures
    cpu **
    on gpu_loc (input |-> sx) **
    (exists* (sy : EM.chest2 f32 b d).
       on gpu_loc (output |-> sy) **
       pure (cumsum_exclusive_post b d sx sy))
{
  scan1d_exclusive_rowblock #f32 cmonoid_fadd_f32 b d
    #(l2_row_major b d) #_
    #(l2_row_major b d) #_
    input output;
  Classical.forall_intro_2
    (Classical.move_requires_2
       (cell_post_eq #b #d (reveal sx)));
  ()
}
#pop-options

(* z3rlimit > 40: top-level coercion of the impl to the interface type,
 * mirrors the inclusive [cumsum_fw_f32]. *)
#push-options "--z3rlimit 100"
let cumsum_exclusive_fw_f32 : cumsum_exclusive_fw_ty f32 = cumsum_exclusive_fw_f32_impl
#pop-options
