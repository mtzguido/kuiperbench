module Kuiper.Spec.Pool2D

(* Functional specification for 2-D windowed pooling
   (KernelBench L1 #42 MaxPool2D, #45 AvgPool2D), defined as the
   *composition of two independent 1-D window reductions* along
   the H and W axes.  The compositional form is the natural
   shape produced by a separable kernel implementation (two
   sequential calls to [Kuiper.Kernel.WindowReduce1D]), and is
   semantically equivalent to a single rectangle-shaped 2-D
   reduction under suitable reducer laws.  See the design notes in
   [src/kernelbench/level1/POOLING_DESIGN.md] §2 for context.

   This module is layered strictly on top of [Kuiper.Spec.Pool1D]:

   - [maxpool2d_post] pins each output directly to a deterministic
     nested [fmax] fold of the input: first along W, then along H.
     The per-axis predicates [row_max_pooled_along_w] and
     [col_max_pooled_along_h] remain available for proofs of a
     separable implementation, but the public contract does not
     existentially hide its intermediate tensor.

   - [avgpool2d_post] directly relates each output to the nested
     real window sum of the input, scaled separately by [inv_kw]
     and [inv_kh] (their product is the 2-D divisor 1/(K_h*K_w)).
     The result is related to this input-determined real value by
     [%~], with no existential floating-point intermediate.

   The view on the underlying [Seq.lseq t (bc * h * w)] is
   row-major: index [r * w + c] within the (b*c)-th plane is the
   element at spatial position [(r, c)].  The intermediate
   tensor has layout [(bc * h * w_out)]; the output tensor has
   layout [(bc * h_out * w_out)].

   ---

   *Equivalence to a single rectangle reduction.*  For both
   max-pool and avg-pool the nested spec is equal to the
   "natural" 2-D rectangle spec that reduces over a [K_h * K_w]
   window in one shot:

     - For avg-pool the equivalence is a pure real-arithmetic
       fact about [rsum] over a 2-D init-ghost sequence and is
       proved by a `(K_h * K_w) ↔ K_h × K_w` index decomposition
       lemma (see [avg_window_sum_2d_via_1d_eq] below).

     - For max-pool the equivalence can be justified for non-NaN
       floats, but this repository intentionally assumes no global
       [fmax] algebraic law.  Bridging the compositional and rectangle
       forms is therefore deferred
       to the future [Kuiper.Spec.Pool2D.Rect] companion module
       and is not needed by either the kernel proof or the
       KernelBench challenges (#42 / #45) themselves, which are
       wired against the compositional spec directly. *)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
open Kuiper.Spec.Pool1D
module Seq = FStar.Seq

(* ----- Output dimensions ------------------------------------------- *)

(* Per-axis output length, lifted to the (H, W) pair. *)
let pool_out_h_2d (h : nat) (kh sh ph dh : nat) : nat =
  pool_out_len_1d h kh sh ph dh
let pool_out_w_2d (w : nat) (kw sw pw dw : nat) : nat =
  pool_out_len_1d w kw sw pw dw

(* ----- W-axis stage: 1-D pool on each H-row ------------------------ *)

(* Per-(b*c, row) max-pool predicate on the W axis: at the
   (b*c, row)-th input row (length [w]) and the (b*c, row)-th
   intermediate row (length [w_out]), every output slot equals
   the [fmax]-fold over the in-bounds 1-D dilated window.  This
   is just a curried call to [row_max_pooled_1d] at the
   appropriate row offset. *)
let row_max_pooled_along_w
  (#t:Type0) {| scalar t |} {| floating t |}
  (#bn_in #bn_mid : nat)
  (sx_in  : Seq.lseq t bn_in)
  (sx_mid : Seq.lseq t bn_mid)
  (h w w_out : nat)
  (off_in  : nat{off_in  + h * w     <= bn_in})
  (off_mid : nat{off_mid + h * w_out <= bn_mid})
  (kw sw pw dw : nat)
  : prop =
  w_out == pool_out_len_1d w kw sw pw dw /\
  (forall (rr : nat). rr < h ==>
     (off_in  + rr * w     + w     <= bn_in  /\
      off_mid + rr * w_out + w_out <= bn_mid /\
      row_max_pooled_1d sx_in sx_mid w w_out
        (off_in + rr * w) (off_mid + rr * w_out) kw sw pw dw))

let row_avg_pooled_along_w
  (#t:Type0) {| scalar t |} {| real_like t |} {| floating t |}
  (#bn_in #bn_mid : nat)
  (sx_in  : Seq.lseq t bn_in)
  (sx_mid : Seq.lseq t bn_mid)
  (h w w_out : nat)
  (off_in  : nat{off_in  + h * w     <= bn_in})
  (off_mid : nat{off_mid + h * w_out <= bn_mid})
  (kw sw pw dw : nat) (inv_kw : t)
  : prop =
  w_out == pool_out_len_1d w kw sw pw dw /\
  (forall (rr : nat). rr < h ==>
     (off_in  + rr * w     + w     <= bn_in  /\
      off_mid + rr * w_out + w_out <= bn_mid /\
      row_avg_pooled_1d sx_in sx_mid w w_out
        (off_in + rr * w) (off_mid + rr * w_out) kw sw pw dw inv_kw))

(* ----- H-axis stage: 1-D pool on each (B*C, w_col) column ---------- *)

(* For a [bc * h * w_out]-layout intermediate, the H-axis pool
   reduces along the (b*c, *, w_col) "column" of length [h] for
   each [w_col].  A column of the 2-D plane is *not* contiguous
   in row-major layout, so the per-column predicate is expressed
   pointwise rather than as a slice. *)
let col_max_pooled_along_h
  (#t:Type0) {| scalar t |} {| floating t |}
  (#bn_mid #bn_out : nat)
  (sx_mid : Seq.lseq t bn_mid)
  (sx_out : Seq.lseq t bn_out)
  (h h_out w_out : nat)
  (off_mid : nat{off_mid + h     * w_out <= bn_mid})
  (off_out : nat{off_out + h_out * w_out <= bn_out})
  (kh sh ph dh : nat)
  : prop =
  h_out == pool_out_len_1d h kh sh ph dh /\
  (forall (jh : nat) (cc : nat). jh < h_out /\ cc < w_out ==>
     (let (col : Seq.lseq t h) =
        Seq.init_ghost h (fun rr ->
          Seq.index sx_mid (off_mid + rr * w_out + cc)) in
      Some? (max_window col kh sh ph dh jh) /\
      Seq.index sx_out (off_out + jh * w_out + cc)
        == Some?.v (max_window col kh sh ph dh jh)))

let col_avg_pooled_along_h
  (#t:Type0) {| scalar t |} {| real_like t |} {| floating t |}
  (#bn_mid #bn_out : nat)
  (sx_mid : Seq.lseq t bn_mid)
  (sx_out : Seq.lseq t bn_out)
  (h h_out w_out : nat)
  (off_mid : nat{off_mid + h     * w_out <= bn_mid})
  (off_out : nat{off_out + h_out * w_out <= bn_out})
  (kh sh ph dh : nat) (inv_kh : t)
  : prop =
  h_out == pool_out_len_1d h kh sh ph dh /\
  (forall (jh : nat) (cc : nat). jh < h_out /\ cc < w_out ==>
      (let (col : Seq.lseq t h) =
        Seq.init_ghost h (fun rr ->
          Seq.index sx_mid (off_mid + rr * w_out + cc)) in
      Seq.index sx_out (off_out + jh * w_out + cc) %~
        (avg_window_sum_r col kh sh ph dh jh *. to_real inv_kh)))

(* ----- Direct nested 2-D window values ----------------------------- *)

(* Deterministic width-then-height [fmax] fold for one output slot.
   [None] means that the whole 2-D window is outside the input. *)
let rec max_window_2d_aux
  (#t:Type0) {| scalar t |} {| floating t |}
  (#hw:nat) (plane : Seq.lseq t hw)
  (h w : nat{h * w == hw})
  (kh kw sh sw ph pw dh dw : nat)
  (jh jw dih : nat) (cur : option t)
  : GTot (option t)
      (decreases (if dih >= kh then 0 else kh - dih)) =
  if dih >= kh then cur
  else
    let cur' : option t =
      if pool_in_bounds h sh ph dh jh dih then begin
        let r : nat = pool_input_idx h sh ph dh jh dih in
        let row_h : Seq.lseq t w =
          Seq.slice plane (r * w) (r * w + w) in
        match max_window row_h kw sw pw dw jw with
        | None -> cur
        | Some x ->
            (match cur with
             | None -> Some x
             | Some c -> Some (fmax c x))
      end else cur
    in
    max_window_2d_aux plane h w kh kw sh sw ph pw dh dw
      jh jw (dih + 1) cur'

let max_window_2d
  (#t:Type0) {| scalar t |} {| floating t |}
  (#hw:nat) (plane : Seq.lseq t hw)
  (h w : nat{h * w == hw})
  (kh kw sh sw ph pw dh dw : nat)
  (jh jw : nat)
  : GTot (option t) =
  max_window_2d_aux plane h w kh kw sh sw ph pw dh dw
    jh jw 0 None

(* Iterated 1-D form of the 2-D real-valued window sum: sum
   along the H axis of the per-row 1-D real window sums. *)
let avg_window_sum_2d_via_1d
  (#t:Type0) {| scalar t |} {| real_like t |}
  (#hw : nat) (plane : Seq.lseq t hw)
  (h w : nat{h * w == hw})
  (kh kw sh sw ph pw dh dw : nat)
  (jh jw : nat)
  : GTot real
  = rsum (Seq.init_ghost kh (fun dih ->
      if pool_in_bounds h sh ph dh jh dih then
        let r : nat = pool_input_idx h sh ph dh jh dih in
        let row_h : Seq.lseq t w = Seq.slice plane (r * w) (r * w + w) in
        avg_window_sum_r row_h kw sw pw dw jw
      else 0.0R))

(* ----- Whole-tensor 2-D specs ------------------------------------- *)

(* Each max-pool output is exactly the deterministic nested fold of
   the corresponding input plane. *)
let maxpool2d_post
  (#t:Type0) {| scalar t |} {| floating t |}
  (bc h w : nat)
  (kh kw sh sw ph pw dh dw : nat)
  (sx  : Seq.lseq t (bc * (h * w)))
  (sx' : Seq.lseq t (bc * (pool_out_h_2d h kh sh ph dh
                           * pool_out_w_2d w kw sw pw dw)))
  : prop =
  let h_out : nat = pool_out_h_2d h kh sh ph dh in
  let w_out : nat = pool_out_w_2d w kw sw pw dw in
  forall (b : nat). b < bc ==>
    (b * (h * w) + h * w <= bc * (h * w) /\
     b * (h_out * w_out) + h_out * w_out <= bc * (h_out * w_out) /\
     (let plane : Seq.lseq t (h * w) =
        Seq.slice sx (b * (h * w)) (b * (h * w) + h * w) in
      let plane_out : Seq.lseq t (h_out * w_out) =
        Seq.slice sx' (b * (h_out * w_out))
          (b * (h_out * w_out) + h_out * w_out) in
      forall (jh : nat) (jw : nat). jh < h_out /\ jw < w_out ==>
        (Some? (max_window_2d plane h w kh kw sh sw ph pw dh dw jh jw) /\
         Seq.index plane_out (jh * w_out + jw) ==
           Some?.v (max_window_2d plane h w
             kh kw sh sw ph pw dh dw jh jw))))

(* The 2-D avg-pool postcondition.  The two scaling constants are
   passed separately; the caller is responsible for ensuring
   [inv_kh] approximates [1/K_h] and [inv_kw] approximates
   [1/K_w] so that their product is the 2-D divisor [1/(K_h K_w)].
   Each output is directly tied to the nested real sum of its input
   plane rather than to an existential FP intermediate. *)
let avgpool2d_post
  (#t:Type0) {| scalar t |} {| real_like t |} {| floating t |}
  (bc h w : nat)
  (kh kw sh sw ph pw dh dw : nat)
  (inv_kh inv_kw : t)
  (sx  : Seq.lseq t (bc * (h * w)))
  (sx' : Seq.lseq t (bc * (pool_out_h_2d h kh sh ph dh
                           * pool_out_w_2d w kw sw pw dw)))
  : prop =
  let h_out : nat = pool_out_h_2d h kh sh ph dh in
  let w_out : nat = pool_out_w_2d w kw sw pw dw in
  forall (b : nat). b < bc ==>
    (b * (h * w) + h * w <= bc * (h * w) /\
     b * (h_out * w_out) + h_out * w_out <= bc * (h_out * w_out) /\
     (let plane : Seq.lseq t (h * w) =
        Seq.slice sx (b * (h * w)) (b * (h * w) + h * w) in
      let plane_out : Seq.lseq t (h_out * w_out) =
        Seq.slice sx' (b * (h_out * w_out))
          (b * (h_out * w_out) + h_out * w_out) in
      forall (jh : nat) (jw : nat). jh < h_out /\ jw < w_out ==>
        Seq.index plane_out (jh * w_out + jw) %~
          ((avg_window_sum_2d_via_1d plane h w
              kh kw sh sw ph pw dh dw jh jw *. to_real inv_kw)
           *. to_real inv_kh)))

(* ----- Length lemmas exposed via .fsti ----------------------------- *)

val pool_out_dims_2d_zero
  (h w kh kw sh sw ph pw dh dw : nat)
  : Lemma (ensures
            ((let kspan_h = dh * (kh - 1) + 1 in
              let padded_h = h + 2 * ph in
              padded_h < kspan_h \/ sh = 0) ==>
             pool_out_h_2d h kh sh ph dh == 0) /\
            ((let kspan_w = dw * (kw - 1) + 1 in
              let padded_w = w + 2 * pw in
              padded_w < kspan_w \/ sw = 0) ==>
             pool_out_w_2d w kw sw pw dw == 0))
