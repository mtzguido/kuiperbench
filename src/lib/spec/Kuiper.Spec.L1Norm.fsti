module Kuiper.Spec.L1Norm

(* Functional specification for row-wise L1 normalisation
   (KernelBench L1 #38):

       y[r, j] = x[r, j] * D / sum_k |x[r, k]|        for each row r.

   This is the PyTorch L1Norm contract: y = x / mean(|x|, dim=1) =
   x * D / sum(|x|, dim=1).

   The input is a (B, D) row-major matrix, modelled directly as an
   [EM.chest2 t B D] (NOT a flattened sequence): the postcondition is
   stated over the matrix entries [acc2 sx r j].

   Per-row spec is structurally analogous to RMSNorm: each row of the
   output is a uniform scaling of the corresponding input row by a
   single per-row factor [scale_r], with [scale_r == div dim_f
   sum_abs_r] and [sum_abs_r] approximating the row's real-valued L1
   norm.  Both [scale_r] and [sum_abs_r] are existentially bound per
   row because the device-side reduction only approximates the real
   sum and [div] is opaque to the spec.

   The L1 reduction is naturally "sum of absolute values".  The kernel
   pushes [fabs] directly into the [HReduce.reduce_batched] pre-map, so
   the device-side per-row reduction is the deterministic left-fold
   [row_reduce_partial fabs sx r D].  We spec the inner sum as
   [rsum (to_real_seq (lseq_map fabs row))]: the float values are
   abs-ed and lifted to reals via [to_real], so the bridge lemma
   [row_reduce_partial_fabs_approx] needs only [to_real_ok] (no
   real-side abs axiom).  Mathematically this is exactly the row's
   L1 norm.

   Edge case (all-zero row): sum_abs = 0, scale = D/0 = +inf in
   IEEE-754, output is NaN-filled.  The PyTorch reference has the
   same behaviour. *)

open Kuiper.Common
open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Kernel.HReduce
open Kuiper.Chest
module Seq = FStar.Seq
module KS = Kuiper.Seq.Common
module EM = Kuiper.EMatrix

(* Sum of absolute values of a row, lifted to reals.
   Defined as the sum of [to_real_seq] applied to the elementwise
   absolute value of the row.  Mathematically this is the row's L1
   norm; we do not factor through a real-side abs because
   [Kuiper.Real] does not currently axiomatise one. *)
let l1_sum_r
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#n : nat)
  (s : Seq.lseq t n) : GTot real =
  rsum (to_real_seq (KS.lseq_map (fabs #t) s))

(* Per-row predicate: row [r] of [sx'] is the L1-normalised version
   of row [r] of [sx], with scale = dim_f / row's L1 norm. *)
let row_l1_normalized
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#b #d : nat)
  (dim_f : t)
  (sx sx' : EM.chest2 t b d)
  (r : natlt b)
  : prop =
  exists (scale : t) (sum_abs : t).
    sum_abs %~ l1_sum_r (EM.ematrix_row sx r) /\
    scale == div dim_f sum_abs /\
    (forall (j : nat). j < d ==>
       acc2 sx' r j == mul scale (acc2 sx r j))

(* Whole-tensor spec: every row is L1-normalised with [dim_f]. *)
let l1norm_post
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (b d : nat)
  (dim_f : t)
  (sx sx' : EM.chest2 t b d)
  : prop =
  forall (r : nat). r < b ==> row_l1_normalized dim_f sx sx' r

(* Bridge lemma: the deterministic device-side left-fold of absolute
   values produced by [reduce_batched fabs] approximates the
   real-valued mathematical L1 norm [l1_sum_r].  Used to discharge the
   [sum_abs %~ ...] conjunct of [row_l1_normalized] from the exact
   post-state of [reduce_batched]. *)
val row_reduce_partial_fabs_approx
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : natlt rows)
  : Lemma (row_reduce_partial (fabs #t) sx r cols
           %~ l1_sum_r (EM.ematrix_row sx r))
