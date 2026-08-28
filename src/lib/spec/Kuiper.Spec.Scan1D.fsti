module Kuiper.Spec.Scan1D

(* Functional specification for inclusive / exclusive prefix-scan
   over a sequence under an arbitrary seeded reducer, plus the
   row-batched 2-D postconditions used by the KernelBench L1 scan
   challenges:

       #89  cumsum            — forward inclusive, sum reducer
       #90  cumprod           — forward inclusive, product reducer
       #91  cumsum_reverse    — reverse inclusive, sum reducer
       #92  cumsum_exclusive  — forward exclusive, sum reducer
       #93  masked_cumsum     — forward inclusive, sum reducer
                                with element-wise pre-masking

   All five variants reduce to a single primitive — the
   reducer-parametric inclusive scan [scan_inclusive] — composed
   with a small number of cheap pre-/post-transforms (mask, shift,
   sequence reversal).  The kernel side (deferred — see
   [src/kernelbench/level1/challenge89/STATUS.txt]) ships a single
   block-wide inclusive scan and the wrapper code follows.

   Floating-point note.  The current kernels scan each row sequentially and
   their internal result is the exact implementation-order fold.  No
   floating-point algebraic laws are assumed.  The public [scan2d_*_post]
   predicates use [%~] to relate that fold to the real-arithmetic ideal,
   mirroring [Kuiper.Spec.SumReduceDim] / [Kuiper.Spec.Frobenius]. *)

open Kuiper
open Kuiper.Chest
open Kuiper.Scalars
open Kuiper.Real
open Kuiper.Approximates
open Kuiper.Monoid.Reduce
module Seq = FStar.Seq
module EM  = Kuiper.EMatrix

(* ── Reducer-parametric scans on a single sequence ────────────────────── *)

(* Inclusive prefix at index [i]: fold over [s[0..i+1]] under [m]. *)
let scan_inclusive_at
  (#t:Type) (m : reducer t) (s : Seq.seq t) (i : nat { i < Seq.length s })
  : GTot t
  = red_fold m m.rid (Seq.slice s 0 (i+1))

(* Whole-sequence inclusive scan: same length as [s]. *)
let scan_inclusive
  (#t:Type) (m : reducer t) (s : Seq.seq t)
  : GTot (Seq.lseq t (Seq.length s))
  = Seq.init_ghost (Seq.length s) (scan_inclusive_at m s)

(* Exclusive prefix at index [i]: fold over [s[0..i]] under [m].
   Cell 0 is therefore the reducer seed [m.rid]. *)
let scan_exclusive_at
  (#t:Type) (m : reducer t) (s : Seq.seq t) (i : nat { i < Seq.length s })
  : GTot t
  = red_fold m m.rid (Seq.slice s 0 i)

let scan_exclusive
  (#t:Type) (m : reducer t) (s : Seq.seq t)
  : GTot (Seq.lseq t (Seq.length s))
  = Seq.init_ghost (Seq.length s) (scan_exclusive_at m s)

(* ── Sequence reversal (used for the reverse-scan variant) ─────────────── *)

let seq_rev (#a:Type) (s : Seq.seq a)
  : GTot (Seq.lseq a (Seq.length s))
  = let n = Seq.length s in
    Seq.init_ghost n (fun i -> s @! (n - 1 - i))

(* Reverse inclusive scan: scan from the right.  Equivalent to
   [reverse (scan_inclusive m (reverse s))], matching the PyTorch
   pattern [torch.cumsum(x.flip(d), d).flip(d)] used by #91. *)
let scan_inclusive_reverse
  (#t:Type) (m : reducer t) (s : Seq.seq t)
  : GTot (Seq.lseq t (Seq.length s))
  = seq_rev (scan_inclusive m (seq_rev s))

(* ── Element-wise pre-mask (used for #93) ──────────────────────────────── *)

(* A masked element is replaced by the reducer seed.  For sum this
   maps [false] to [0.0], for product to [1.0]. *)
let mask_step
  (#t:Type) (m : reducer t) (x : t) (b : bool) : t
  = if b then x else m.rid

let mask_seq
  (#t:Type) (m : reducer t)
  (s : Seq.seq t) (mask : Seq.seq bool { Seq.length mask == Seq.length s })
  : GTot (Seq.lseq t (Seq.length s))
  = Seq.init_ghost (Seq.length s) (fun i -> mask_step m (s @! i) (mask @! i))

let scan_inclusive_masked
  (#t:Type) (m : reducer t)
  (s : Seq.seq t) (mask : Seq.seq bool { Seq.length mask == Seq.length s })
  : GTot (Seq.lseq t (Seq.length s))
  = scan_inclusive m (mask_seq m s mask)

(* ── Trivial structural lemmas ─────────────────────────────────────────── *)

(* The output is a transform of the input of the *same* length;
   expose this as an SMTPat so callers do not have to chase it
   manually.  Each follows by [Seq.init_ghost]'s defining
   length-equation. *)

val scan_inclusive_length
  (#t:Type) (m : reducer t) (s : Seq.seq t)
  : Lemma (Seq.length (scan_inclusive m s) == Seq.length s)
          [SMTPat (Seq.length (scan_inclusive m s))]

val scan_exclusive_length
  (#t:Type) (m : reducer t) (s : Seq.seq t)
  : Lemma (Seq.length (scan_exclusive m s) == Seq.length s)
          [SMTPat (Seq.length (scan_exclusive m s))]

val seq_rev_length
  (#a:Type) (s : Seq.seq a)
  : Lemma (Seq.length (seq_rev s) == Seq.length s)
          [SMTPat (Seq.length (seq_rev s))]

(* Index laws — direct from the [init_ghost] body.  Useful when a
   client wants the per-cell defining equation without unfolding. *)

val scan_inclusive_index
  (#t:Type) (m : reducer t) (s : Seq.seq t) (i : nat { i < Seq.length s })
  : Lemma ((scan_inclusive m s) @! i == scan_inclusive_at m s i)
          [SMTPat ((scan_inclusive m s) @! i)]

val scan_exclusive_index
  (#t:Type) (m : reducer t) (s : Seq.seq t) (i : nat { i < Seq.length s })
  : Lemma ((scan_exclusive m s) @! i == scan_exclusive_at m s i)
          [SMTPat ((scan_exclusive m s) @! i)]

(* Boundary value of the exclusive scan: cell 0 is the reducer seed.
   This comes free from [Seq.slice s 0 0 == Seq.empty] and
   [seq_fold_left _ acc empty == acc]. *)
val scan_exclusive_zero
  (#t:Type) (m : reducer t) (s : Seq.seq t { Seq.length s > 0 })
  : Lemma ((scan_exclusive m s) @! 0 == m.rid)

(* ── Row-batched 2-D postconditions ────────────────────────────────────── *)

(* The KernelBench inputs are (batch, n) with the scan along the
   last (inner) axis.  The 2-D chest2 view is therefore directly
   the input shape with [n_rows = batch], [n_cols = n].  Each
   output row is the scan of the corresponding input row, lifted
   to the real-arithmetic ideal scan via [%~]. *)

(* #89 / #90: forward inclusive scan along the inner dimension. *)
let scan2d_inclusive_post
  (#t:Type0) {| scalar t, real_like t |}
  (m_r : reducer real)
  (n_rows n_cols : nat)
  (sx : chest2 t n_rows n_cols)
  (sy : chest2 t n_rows n_cols)
  : prop
  = forall (r : nat) (i : nat).
        r < n_rows /\ i < n_cols ==>
        acc2 sy r i %~
          scan_inclusive_at m_r (to_real_seq (EM.ematrix_row sx r)) i

(* #92: forward exclusive. *)
let scan2d_exclusive_post
  (#t:Type0) {| scalar t, real_like t |}
  (m_r : reducer real)
  (n_rows n_cols : nat)
  (sx : chest2 t n_rows n_cols)
  (sy : chest2 t n_rows n_cols)
  : prop
  = forall (r : nat) (i : nat).
        r < n_rows /\ i < n_cols ==>
        acc2 sy r i %~
          scan_exclusive_at m_r (to_real_seq (EM.ematrix_row sx r)) i

(* #91: reverse inclusive — scan of the reversed row, then reversed.
   Equivalently: cell [i] is the fold of [row[i..n_cols]]. *)
let scan2d_inclusive_reverse_post
  (#t:Type0) {| scalar t, real_like t |}
  (m_r : reducer real)
  (n_rows n_cols : nat)
  (sx : chest2 t n_rows n_cols)
  (sy : chest2 t n_rows n_cols)
  : prop
  = forall (r : nat) (i : nat).
        r < n_rows /\ i < n_cols ==>
        acc2 sy r i %~
          (scan_inclusive_reverse m_r (to_real_seq (EM.ematrix_row sx r)) @! i)

(* #93: forward inclusive of the masked row.  [smask] is the
   batched boolean mask, same shape as the input. *)
let scan2d_inclusive_masked_post
  (#t:Type0) {| scalar t, real_like t |}
  (m_r : reducer real)
  (n_rows n_cols : nat)
  (sx : chest2 t n_rows n_cols)
  (smask : chest2 bool n_rows n_cols)
  (sy : chest2 t n_rows n_cols)
  : prop
  = forall (r : nat) (i : nat).
        r < n_rows /\ i < n_cols ==>
        acc2 sy r i %~
          (scan_inclusive_masked m_r
              (to_real_seq (EM.ematrix_row sx r))
              (EM.ematrix_row smask r)
            @! i)
