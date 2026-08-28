module Kuiper.Spec.Pool1D

(* Functional specification for 1-D windowed pooling
   (KernelBench L1 #41 MaxPool1D, #44 AvgPool1D, and the spine of
   the future 2-D / 3-D pooling kernels #42/#43/#45/#46, which all
   reduce to a stack of 1-D window reductions along independent
   spatial axes).

   The host-side view is an Array2 with rows = B*C and cols = L
   (sequence length).  Each output position [j] of an output row
   reduces an in-bounds dilated window of K input positions, with
   stride S, padding P (zero / -inf depending on the reducer), and
   dilation D:

       in_pos(j, di) = j*S + di*D - P     for di in [0, K)

   The output row length is the standard PyTorch formula

       L_out = floor((L + 2P - D*(K - 1) - 1) / S) + 1

   and is computed by [pool_out_len_1d] below.

   Two reductions are specified here:

   - **Max-pool (#41)** is *exact*: the spec pins the output to a
     deterministic, increasing-offset [fmax]-fold over the in-bounds
     elements of each window (no existential, no [%~]).
     Out-of-bounds (padded) positions are *skipped* — this matches
     PyTorch's effective semantics, which uses -inf padding so
     that padded positions never win the max.  We do not model
     the -inf explicitly because [floating] does not currently
     expose a sentinel scalar; instead we carry an [option t]
     accumulator and require the caller-visible postcondition to
     consume a [Some] (windows that are fully out-of-bounds —
     which require pathological dimensions — are ill-formed).

   - **Avg-pool (#44)** is *approximate*: floating-point sum is
     non-associative, so the spec uses the [%~] approximate-post
     pattern (mirrored on Frobenius / L2Norm).  The window is
     summed over reals with out-of-bounds positions contributing
     0 (matching PyTorch's default [count_include_pad=True]), and
     the floating-point output is required to equal the FP sum
     rescaled by a caller-provided [inv_k] of type [t].  The
     caller is responsible for [inv_k ≈ 1 / K] (typically by
     building it as [div one count_t] where [count_t %~ K]).

   This 1-D spec is the building block for the higher-dimensional
   pooling kernels: 2-D pool over [(H, W)] is a 1-D row-wise pool
   over [W] composed with a 1-D row-wise pool over [H] (after
   transposition / row-major reshape), and similarly for 3-D.
   The corresponding 2-D / 3-D specs will be defined as folds of
   this one. *)

open Kuiper.Real
open Kuiper.Scalars
open Kuiper.Floating.Base
open Kuiper.Approximates
module Seq = FStar.Seq

(* ----- Output length formula --------------------------------------- *)

(* PyTorch-compatible 1-D pool output length, in [floor] (non-ceil)
   mode.  Returns 0 when the dilated kernel cannot fit in the padded
   input (a degenerate case the caller should avoid).

   K must be >= 1, S must be >= 1, D must be >= 1.  We do not refine
   the arguments here so that the spec accepts the natural [nat]
   parameters of the kernel; callers are expected to pass sane
   dimensions. *)
let pool_out_len_1d (l k s p d : nat) : nat =
  let kspan = d * (k - 1) + 1 in           (* dilated window size *)
  let padded = l + 2 * p in
  if padded < kspan || s = 0 then 0
  else (padded - kspan) / s + 1

(* ----- Windowed indexing ------------------------------------------- *)

(* Whether output position [j], kernel offset [di] lands inside the
   physical input row [0, l). *)
let pool_in_bounds (l s p d : nat) (j di : nat) : bool =
  let raw : int = j * s + di * d - p in
  raw >= 0 && raw < l

(* Physical input index for an in-bounds (j, di).  The refinement on
   the result is what makes this safe to use as a [Seq.index]
   argument. *)
let pool_input_idx (l s p d : nat) (j di : nat)
  : Pure nat
      (requires pool_in_bounds l s p d j di)
      (ensures fun i -> i < l) =
  let raw : int = j * s + di * d - p in
  raw

(* ----- Avg-pool: real-valued window sum ---------------------------- *)

(* Real-valued sum of the K-element dilated window centered at output
   position [j] of input row [row], with out-of-bounds positions
   contributing 0.0R.  This matches PyTorch's [count_include_pad=True]
   convention used by [nn.AvgPool{1,2,3}d] in their default
   configuration.  The divisor is left to the caller (typically [K]
   exactly, regardless of how many positions are in-bounds). *)
let avg_window_sum_r
  (#t:Type0) {| scalar t |} {| real_like t |}
  (#l:nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat) : GTot real =
  rsum (Seq.init_ghost k (fun di ->
    if pool_in_bounds l s p d j di
    then to_real (Seq.index row (pool_input_idx l s p d j di))
    else 0.0R))

(* ----- Max-pool: FP fmax fold over in-bounds positions ------------- *)

(* Fold [fmax] over the in-bounds elements of the K-element window
   at output position [j], starting from kernel offset [di] with
   accumulator [cur].  Returns [None] iff every position [di..k) is
   out-of-bounds AND [cur] is [None].  In all the pooling
   configurations that arise from KernelBench (#41-#46) at least one
   position per output slot is in-bounds, so the caller-visible
   spec consumes [Some _]. *)
let rec window_fmax_aux
  (#t:Type0) {| scalar t |} {| floating t |}
  (#l:nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat) (di : nat) (cur : option t)
  : GTot (option t) (decreases (if di >= k then 0 else k - di)) =
  if di >= k then cur
  else
    let cur' : option t =
      if pool_in_bounds l s p d j di then begin
        let v : t = Seq.index row (pool_input_idx l s p d j di) in
        match cur with
        | None    -> Some v
        | Some c  -> Some (fmax c v)
      end else cur
    in
    window_fmax_aux row k s p d j (di + 1) cur'

let max_window
  (#t:Type0) {| scalar t |} {| floating t |}
  (#l:nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat)
  : GTot (option t) =
  window_fmax_aux row k s p d j 0 None

(* ----- Per-row predicates ------------------------------------------ *)

(* Per-row max-pool predicate: at the row starting at [off_in] in
   [sx] (length [l]) and the row starting at [off_out] in [sx']
   (length [lo == pool_out_len_1d l k s p d]), every output slot
   equals the [fmax]-fold over the in-bounds 1-D dilated window in
   the input row.  Exact equality records this specified fold order and
   does not assume an algebraic law for [fmax]. *)
let row_max_pooled_1d
  (#t:Type0) {| scalar t |} {| floating t |}
  (#bn_in #bn_out : nat)
  (sx  : Seq.lseq t bn_in)
  (sx' : Seq.lseq t bn_out)
  (l lo : nat)
  (off_in  : nat{off_in  + l  <= bn_in})
  (off_out : nat{off_out + lo <= bn_out})
  (k s p d : nat)
  : prop =
  lo == pool_out_len_1d l k s p d /\
  (let row_in  : Seq.lseq t l  = Seq.slice sx  off_in  (off_in  + l)  in
   let row_out : Seq.lseq t lo = Seq.slice sx' off_out (off_out + lo) in
   forall (j : nat). j < lo ==>
     (Some? (max_window row_in k s p d j) /\
      Seq.index row_out j == Some?.v (max_window row_in k s p d j)))

(* Per-row avg-pool predicate: every output slot in the [lo]-length
   row at [off_out] of [sx'] approximates the real-valued windowed
   average of the [l]-length row at [off_in] of [sx], scaled by
   caller-supplied [inv_k] (which the caller is responsible for
   relating to [1/K]). *)
let row_avg_pooled_1d
  (#t:Type0) {| scalar t |} {| real_like t |} {| floating t |}
  (#bn_in #bn_out : nat)
  (sx  : Seq.lseq t bn_in)
  (sx' : Seq.lseq t bn_out)
  (l lo : nat)
  (off_in  : nat{off_in  + l  <= bn_in})
  (off_out : nat{off_out + lo <= bn_out})
  (k s p d : nat)
  (inv_k : t)
  : prop =
  lo == pool_out_len_1d l k s p d /\
  (let row_in  : Seq.lseq t l  = Seq.slice sx  off_in  (off_in  + l)  in
   let row_out : Seq.lseq t lo = Seq.slice sx' off_out (off_out + lo) in
   forall (j : nat). j < lo ==>
     (exists (sm : t).
       sm %~ avg_window_sum_r row_in k s p d j /\
       Seq.index row_out j == mul sm inv_k))

(* ----- Whole-tensor specs ------------------------------------------ *)

(* The host-side view of the input is an Array2 of shape
   (B*C, L); the output has shape (B*C, pool_out_len_1d L K S P D).
   Both are flattened to row-major [Seq.lseq t (bc * L)] and
   [Seq.lseq t (bc * lo)] respectively. *)

let maxpool1d_post
  (#t:Type0) {| scalar t |} {| floating t |}
  (bc l : nat) (k s p d : nat)
  (sx  : Seq.lseq t (bc * l))
  (sx' : Seq.lseq t (bc * pool_out_len_1d l k s p d))
  : prop =
  let lo : nat = pool_out_len_1d l k s p d in
  forall (r : nat). r < bc ==>
    (r * l + l <= bc * l /\ r * lo + lo <= bc * lo /\
     row_max_pooled_1d sx sx' l lo (r * l) (r * lo) k s p d)

let avgpool1d_post
  (#t:Type0) {| scalar t |} {| real_like t |} {| floating t |}
  (bc l : nat) (k s p d : nat) (inv_k : t)
  (sx  : Seq.lseq t (bc * l))
  (sx' : Seq.lseq t (bc * pool_out_len_1d l k s p d))
  : prop =
  let lo : nat = pool_out_len_1d l k s p d in
  forall (r : nat). r < bc ==>
    (r * l + l <= bc * l /\ r * lo + lo <= bc * lo /\
     row_avg_pooled_1d sx sx' l lo (r * l) (r * lo) k s p d inv_k)

(* ----- Length lemma exposed via .fsti ------------------------------ *)

val pool_out_len_1d_bound
  (l k s p d : nat)
  : Lemma (ensures
            (let kspan = d * (k - 1) + 1 in
             let padded = l + 2 * p in
             (padded < kspan || s = 0) ==> pool_out_len_1d l k s p d == 0))
