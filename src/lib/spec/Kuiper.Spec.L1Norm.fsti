module Kuiper.Spec.L1Norm

(* Functional specification for row-wise L1 normalisation
   (KernelBench L1 #38):

       y[r, j] = x[r, j] * D / sum_k |x[r, k]|        for each row r.

   This is the PyTorch L1Norm contract: y = x / mean(|x|, dim=1) =
   x * D / sum(|x|, dim=1).

   The input is a (B, D) row-major matrix, modelled directly as an
   [chest2 t B D] (NOT a flattened sequence): the postcondition is
   stated over the matrix entries [acc2 sx r j].

   The public per-row spec directly relates each output element to an
   explicitly supplied real input scaled by [D / sum |x|].  The floating
   reduction and division values remain proof-local.

   The pinned floating interface does not give the primitive [fabs] a real
   approximation law.  The kernel therefore uses the equivalent branchless
   expression [fmax x (0 - x)].  Its packaged [sub_approx] and [fmax_approx]
   laws prove that it approximates [rmax x (0 - x)], the real absolute value,
   without a local axiom.

   Rows whose real L1 sum is zero are outside [l1norm_domain], so the real
   contract intentionally makes no claim about their NaN-producing IEEE-754
   execution. *)

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

(* Floating and real absolute-value expressions connected solely by packaged
   arithmetic approximation laws. *)
inline_for_extraction
let l1_abs (#t:Type0) {| scalar t, floating t |} (x : t) : t =
  fmax x (sub zero x)

let l1_abs_r (x : real) : real =
  rmax x (0.0R -. x)

let l1_sum_r (#n : nat) (s : Seq.lseq real n) : GTot real =
  rsum (KS.lseq_map l1_abs_r s)

let l1_scale_r (#d:pos) (row : Seq.lseq real d)
  : real =
  let s = l1_sum_r row in
  if s =!= 0.0R then FStar.Real.of_int d /. s else 0.0R

let l1norm_domain (#b:nat) (#d:pos) (rx:chest2 real b d) : prop =
  forall (r:nat). r < b ==> l1_sum_r (EM.ematrix_row rx r) =!= 0.0R

(* Per-row predicate: every output directly approximates the
   mathematical L1-normalized input cell. *)
let row_l1_normalized
  (#b : nat) (#d : pos)
  (rx : chest2 real b d)
  (sx' : chest2 f32 b d)
  (r : natlt b)
  : prop =
  let row = EM.ematrix_row rx r in
  forall (j : nat). j < d ==>
    acc2 sx' r j %~ (acc2 rx r j *. l1_scale_r row)

(* Whole-tensor spec: every row is L1-normalised with [dim_f]. *)
let l1norm_post
  (b : nat) (d : pos)
  (rx : chest2 real b d)
  (sx' : chest2 f32 b d)
  : prop =
  forall (r : nat). r < b ==> row_l1_normalized rx sx' r

(* Bridge lemma from a floating row approximating an explicit real row to the
   real L1 sum. *)
val row_reduce_partial_l1_abs_approx
  (#t:Type0)
  {| scalar t, real_like t, floating t, floating_real_like t |}
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (rx : chest2 real rows cols { sx %~ rx })
  (r : natlt rows)
  : Lemma (row_reduce_partial (l1_abs #t) sx r cols
           %~ l1_sum_r (EM.ematrix_row rx r))
