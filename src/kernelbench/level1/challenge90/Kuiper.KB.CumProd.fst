module Kuiper.KB.CumProd

#lang-pulse
open Kuiper
open Kuiper.KB.Compat.Product
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Spec.Scan1D
open Kuiper.Monoid.Reduce
open Kuiper.Monoid.Reduce.F32
open Kuiper.Kernel.Scan1D.RowBlock
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

(* Per-cell bridge from the bit-exact f32 scan postcondition to the
   [%~]-approximated real-arithmetic prefix product. *)
let macc_scan2d_inclusive_result_f32_mul
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows) (i : natlt cols)
  : Lemma (acc2 (scan2d_inclusive_result cmonoid_fmul_f32 sx) r i
           == scan_inclusive_at cmonoid_fmul_f32 (EM.ematrix_row sx r) i)
  = ()

#push-options " --z3rlimit 60"
let cell_post_eq
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (i : natlt cols)
  : Lemma
      (acc2 (scan2d_inclusive_result cmonoid_fmul_f32 sx) r i
       %~ rprod (Seq.slice (to_real_seq (EM.ematrix_row sx r)) 0 (i + 1)))
  = let row : Seq.seq f32 = EM.ematrix_row sx r in
    let s  : Seq.seq f32 = Seq.slice row 0 (i + 1) in
    let s' : Seq.seq real = Seq.slice (to_real_seq row) 0 (i + 1) in
    slice_is_approx #f32 row (i + 1);
    prod_is_approx #f32 s s';
    macc_scan2d_inclusive_result_f32_mul sx r i;
    cmonoid_fmul_f32_proj ();
    assert (acc2 (scan2d_inclusive_result cmonoid_fmul_f32 sx) r i
              == scan_inclusive_at cmonoid_fmul_f32 row i);
    assert (scan_inclusive_at cmonoid_fmul_f32 row i
              == SC.seq_fold_left mul one s);
    assert (SC.seq_fold_left mul one s %~ rprod s')
#pop-options

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn cumprod_fw_f32_impl
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
       pure (cumprod_post b d sx sy))
{
  scan1d_inclusive_rowblock #f32 cmonoid_fmul_f32 b d
    #(l2_row_major b d) #_
    #(l2_row_major b d) #_
    input output;
  Classical.forall_intro_2
    (Classical.move_requires_2
       (cell_post_eq #b #d (reveal sx)));
  ()
}
#pop-options

#push-options "--z3rlimit 100"
let cumprod_fw_f32 : cumprod_fw_ty f32 = cumprod_fw_f32_impl
#pop-options
