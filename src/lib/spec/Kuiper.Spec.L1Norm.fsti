module Kuiper.Spec.L1Norm

(* Functional specification for row-wise L1 normalisation
   (KernelBench L1 #38):

       y[r, j] = x[r, j] * D / sum_k |x[r, k]|        for each row r.

   This is the PyTorch L1Norm contract: y = x / mean(|x|, dim=1) =
   x * D / sum(|x|, dim=1).

   The input is a (B, D) row-major matrix, modelled directly as an
   [chest2 t B D] (NOT a flattened sequence): the postcondition is
   stated over the matrix entries [acc2 sx r j].

   The public per-row spec directly relates each output element to the
   corresponding real input scaled by [D / sum |x|].  The floating
   reduction and division values remain proof-local.

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
unfold let f32 = Kuiper.Float32.t

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

let l1_scale_r (#d:pos) (row : Seq.lseq f32 d)
  : real =
  let s = l1_sum_r row in
  if s =!= 0.0R then FStar.Real.of_int d /. s else 0.0R

let l1norm_domain (#b:nat) (#d:pos) (sx:chest2 f32 b d) : prop =
  forall (r:nat). r < b ==> l1_sum_r (EM.ematrix_row sx r) =!= 0.0R

(* Per-row predicate: every output directly approximates the
   mathematical L1-normalized input cell. *)
let row_l1_normalized
  (#b : nat) (#d : pos)
  (sx : chest2 f32 b d)
  (sx' : chest2 f32 b d)
  (r : natlt b)
  : prop =
  let row = EM.ematrix_row sx r in
  forall (j : nat). j < d ==>
    acc2 sx' r j %~ (to_real (acc2 sx r j) *. l1_scale_r row)

(* Whole-tensor spec: every row is L1-normalised with [dim_f]. *)
let l1norm_post
  (b : nat) (d : pos)
  (sx : chest2 f32 b d)
  (sx' : chest2 f32 b d)
  : prop =
  forall (r : nat). r < b ==> row_l1_normalized sx sx' r

(* Bridge lemma: the deterministic device-side left-fold of absolute
   values produced by [reduce_batched fabs] approximates the
   real-valued mathematical L1 norm [l1_sum_r].  Used to discharge the
   [sum_abs %~ ...] conjunct of [row_l1_normalized] from the exact
   post-state of [reduce_batched]. *)
val row_reduce_partial_fabs_approx
  (#t:Type0) {| scalar t, real_like t, floating t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (r : natlt rows)
  : Lemma (row_reduce_partial (fabs #t) sx r cols
           %~ l1_sum_r (EM.ematrix_row sx r))
