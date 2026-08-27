module Kuiper.Kernel.HReduce.Block

friend Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Tensor
open Kuiper.Chest1.Helpers
open Kuiper.Bijection { ( =~ ), bij_sym }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }
// Re-open after Kuiper.Tensor so the seq-level `@!`/`seq![..]`/`@+` notations
// shadow the shape-indexing `@!` pulled in via Kuiper.Shape.
open Kuiper.Seq.Common
open Kuiper.Kernel.HReduce
open Kuiper.EMatrix
open Kuiper.SHMem
open Pulse.Lib.Send

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module TC = FStar.Tactics.Typeclasses

(* ── Ported SUM seq lemmas ─────────────────────────────────────────────────
   Local copies of the [private] helpers in [Kuiper.Kernel.HReduce] that are
   needed in this module's own proofs (their SMTPats / facts are not visible
   across the [friend] boundary). They are direct ports from the 1D SUM file. *)

(* Per-element step lemma for the strided reduction in [sum_stride_map_2d]:
   appending the [k]-th strided element to the running sum of the first [k]
   elements yields the running sum of the first [k+1] elements. *)
private let rsum_seq_stride_step
  (rs : seq real)
  (stride : pos)
  (off : nat{off < stride})
  (k : nat)
  : Lemma (requires k < seq_stride_length rs stride off /\
                    k * stride + off < Seq.length rs)
          (ensures
            rsum (seq_take k (seq_stride rs stride off)) +. (rs @! (k * stride + off)) ==
            rsum (seq_take (k + 1) (seq_stride rs stride off)))
  = let ss = seq_stride rs stride off in
    let a = seq_take k ss in
    let single = Seq.slice ss k (k+1) in
    let v = rs @! (off + k * stride) in
    Kuiper.Seq.Common.lem_append_slice ss 0 k (k+1);
    assert (Seq.equal (seq_take (k+1) ss) (Seq.append a single));
    assert (Seq.length single == 1);
    assert (Seq.index single 0 == Seq.index ss k);
    assert (Seq.index ss k == v);
    Kuiper.Seq.Common.lem_one_elem single v;
    Kuiper.Approximates.rsum_append a single

// If k <= (n - off + stride - 1) / stride, k * stride + off >= n, and off < stride,
// then k == (n - off + stride - 1) / stride.
private let stride_length_exact (k n stride off : nat)
  : Lemma
      (requires
        stride > 0 /\ off < stride /\
        k <= (n - off + stride - 1) / stride /\
        k * stride + off >= n)
      (ensures k == (n - off + stride - 1) / stride)
  = Math.Lemmas.lemma_div_le (k * stride) (n - off + stride - 1) stride;
    Math.Lemmas.cancel_mul_div k stride

(* [rsum] of a one-element sequence is its element. *)
private let rsum_singleton (x : real)
  : Lemma (rsum (seq![x]) == x)
  = let SCons hd tl = view_seq (seq![x]) in
    assert (Seq.equal tl (Seq.empty #real))

(* Extracted into a clean (quantifier-free) context: the trivial step
   [min (tid + 1) nth == tid + 1] when [tid < nth] is fragile when proved
   inline under the ambient row-approximation quantifiers. *)
let min_tid_pow2_step (tid nth k : nat)
  : Lemma (requires tid < nth /\ pow2 k == 1)
          (ensures min (tid + pow2 k) nth == tid + 1)
  = ()

(* Wrapper around [tensor_read] that takes the row/col already refined as
   [szlt rows]/[szlt cols], so the [conc] index is well-typed from the
   parameter types. *)
inline_for_extraction noextract
fn read_at
  (#et:Type0) {| scalar et |}
  (rows : szp)
  (cols : szp)
  (#lin : layout2 rows cols) {| ctlayout lin |}
  (x : array2 et lin)
  (row : szlt rows)
  (col : szlt cols)
  (#sx : chest2 et rows cols)
  (#f : perm)
  preserves
    x |-> Frac f sx
  returns
    res : et
  ensures
    pure (res == acc2 sx row col)
{
  tensor_read x (cidx2 row col)
}

(* Drop a per-element [pure] clause from a [forevery] predicate. *)
ghost
fn forevery_drop_pure
  (#a:Type0)
  (p : a -> slprop)
  (q : a -> prop)
  requires
    forall+ (x:a). p x ** pure (q x)
  ensures
    forall+ (x:a). p x
{
  forevery_map
    (fun (x:a) -> p x ** pure (q x))
    p
    fn x { drop_ (pure (q x)) }
}

(* Per-thread input-side reduction: like [sum_stride_map] but reads from a
   single row of a 2-D tensor. Handles empty strided buckets (starts at [zero]),
   so no [off < cols] requirement is needed (unlike the MAX analogue). *)
#push-options "--fuel 4 --ifuel 8 --z3rlimit 200"
inline_for_extraction noextract
fn sum_stride_map_2d
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp)
  (cols : szp)
  (#lin : layout2 rows cols) {| ctlayout lin |}
  (x : array2 et lin)
  (row : szlt rows)
  (stride : szp)
  (off : szlt stride)
  (#sx : chest2 et rows cols)
  (vr_row : erased (lseq real cols))
  (#f : perm)
  preserves
    gpu ** x |-> Frac f sx **
    pure (forall (j:nat). j < SZ.v cols ==> acc2 sx row j %~ (vr_row @! j)) **
    pure (SZ.fits (SZ.v cols + stride))
  returns
    res : et
  ensures
    pure (res %~ rsum (seq_stride (lseq_map pre_map_r vr_row) stride off))
{
  let mut acc : et = zero;
  let mut idx : sz = off;
  let gidx = galloc #nat 0;

  while (!idx <^ cols)
    invariant
      live acc ** live gidx ** live idx **
      pure (gread gidx <= seq_stride_length (lseq_map pre_map_r vr_row) stride off /\
            !idx < cols + stride /\
            !acc %~ rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr_row) stride off)) /\
            SZ.v !idx == gread gidx * stride + off
      ) **
      emp
    decreases (cols + stride - !idx)
  {
    assert pure (gread gidx < seq_stride_length vr_row stride off);

    let idx_raw : sz = !idx;
    assert pure (SZ.v idx_raw < SZ.v cols);
    let idx_v : szlt cols = idx_raw;
    let v = read_at rows cols x row idx_v;
    let v' = pre_map v;
    (**)assert (pure (v == acc2 sx row idx_v));
    (**)assert (pure (v %~ (vr_row @! SZ.v idx_v)));
    (**)assert (pure (v' %~ (lseq_map pre_map_r vr_row @! SZ.v idx_v)));

    a_add !acc v'
      (rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_r vr_row) stride off)))
      ((lseq_map pre_map_r vr_row) @! SZ.v idx_v);
    rsum_seq_stride_step (lseq_map pre_map_r vr_row) stride off (gread gidx);

    let vgidx = gread gidx;
    Math.Lemmas.distributivity_add_left vgidx 1 stride;

    acc := !acc `add` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);
    ()
  };

  stride_length_exact (gread gidx) (Seq.length (lseq_map pre_map_r vr_row)) stride off;
  assert pure (gread gidx == seq_stride_length (lseq_map pre_map_r vr_row) stride off);
  assert pure (seq_take (seq_stride_length (lseq_map pre_map_r vr_row) stride off)
                       (seq_stride (lseq_map pre_map_r vr_row) stride off)
              == seq_stride (lseq_map pre_map_r vr_row) stride off);
  drop_ (gidx |-> _);
  !acc
}
#pop-options

(* ── Per-thread predicates for the per-block kernel ────────────────────── *)

unfold
let kpre_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols)
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)) **
    exists* (v : et). tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ()) v

unfold
let kpost_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols)
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    if_ (op_Equals #nat tid 0) (
      live_c_shmem #(SHArray et nth) shmem._1 **
      exists* (v : et).
        Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))
    )

(* ── Per-thread kernel function ────────────────────────────────────────── *)

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn kf_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt rows)
  (tid : szlt nth)
  ()
  requires
    gpu **
    pure (c_shmems_inv shmem) **
    kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1)
                       (vr_partial pre_map_r (ematrix_row vr bid) nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1)
                       (vr_partial pre_map_r (ematrix_row vr bid) nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  unfold kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid tid;

  let (gsa, _) = shmem;
  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  (* Row of vr at bid, as an lseq real cols. *)
  let vr_row : erased (lseq real cols) = hide (ematrix_row (reveal vr) bid);

  (* Bridge from (sx %~ vr) to row-level approximation. *)
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 (vr_row @! j) == acc2 (reveal vr) bid j);
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 acc2 sx bid j %~ (vr_row @! j));

  (* Compute partial sum over stride and write to shmem. *)
  let psum : et = sum_stride_map_2d pre_map pre_map_r rows cols x bid nth tid vr_row;
  tensor_write_cell sa (tid, ()) psum;

  (* Set up tree reduction state. *)
  let mut n : szlt 32 = 0sz;

  let psum_chest : chest1 et 1 = mk1 #et #1 (fun _ -> psum);
  slice_singleton sa tid psum psum_chest;

  (* psum %~ (vr_partial pre_map_r (ematrix_row vr bid) nth) @! tid == rsum (single-element slice [tid, tid+1)) *)
  (**)assert (pure (psum %~ (reveal (vr_partial pre_map_r (ematrix_row vr bid) nth) @! SZ.v tid)));
  (**)rsum_singleton (reveal (vr_partial pre_map_r (ematrix_row vr bid) nth) @! SZ.v tid);
  (**)assert (pure (Seq.equal (Seq.slice (reveal (vr_partial pre_map_r (ematrix_row vr bid) nth)) tid (tid + 1)) (seq![reveal (vr_partial pre_map_r (ematrix_row vr bid) nth) @! SZ.v tid])));
  (**)assert (pure (acc1 psum_chest 0 %~ rsum (Seq.slice (reveal (vr_partial pre_map_r (ematrix_row vr bid) nth)) tid (tid + 1))));
  (**)fold (array1_pts_to_slice_sum sa tid (tid + 1) (vr_partial pre_map_r (ematrix_row vr bid) nth));
  (**)assert (pure (pow2 (SZ.v !n) == 1));
  (**)min_tid_pow2_step tid nth (SZ.v !n);
  (**)rewrite (array1_pts_to_slice_sum sa tid (tid + 1) (vr_partial pre_map_r (ematrix_row vr bid) nth))
  (**)     as (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) (vr_partial pre_map_r (ematrix_row vr bid) nth));
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) (vr_partial pre_map_r (ematrix_row vr bid) nth));

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) (vr_partial pre_map_r (ematrix_row vr bid) nth)) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration nth sa (vr_partial pre_map_r (ematrix_row vr bid) nth) tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 it) nth) (vr_partial pre_map_r (ematrix_row vr bid) nth)))
  as
    (if_ (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial pre_map_r (ematrix_row vr bid) nth)));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  if (tid = 0sz) {
    if_elim_true' (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial pre_map_r (ematrix_row vr bid) nth));
    if_elim_true' (op_Equals #nat tid 0) (Cell output (((SZ.v bid <: natlt rows), ()) <: abs (rows @| INil)) |-> (acc1 sout bid));
    unfold array1_pts_to_slice_sum sa 0 nth (vr_partial pre_map_r (ematrix_row vr bid) nth);
    (**)strided_sum_is_sum pre_map_r vr_row nth;
    (**)assert (pure (Seq.equal (Seq.slice (reveal (vr_partial pre_map_r (ematrix_row vr bid) nth)) 0 nth) (reveal (vr_partial pre_map_r (ematrix_row vr bid) nth))));

    let res = array1_read_from_slice sa 0sz;
    tensor_write_cell output (bid, ()) res;

    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let css : chest1 et nth = hide (mk1 #et #nth (fun (k:natlt nth) -> acc1 ss k));
    forevery_refine_ext' #nat #(fun (k:nat) -> 0 <= k /\ k < nth) (fun (k:nat) -> k < nth) _;
    forevery_ext
      (fun (k:natlt nth) -> tensor_pts_to_cell sa ((k <: natlt nth), ()) (acc1 ss (k - 0)))
      (fun (k:natlt nth) -> tensor_pts_to_cell sa (abs_bij.gg k) (acc (reveal css) (abs_bij.gg k)));
    forevery_iso_back (abs_bij #nth)
      (fun (i : abs (nth @| INil)) -> tensor_pts_to_cell sa i (acc (reveal css) i));
    tensor_implode sa #1.0R #(reveal css);
    rewrite each sa as from_array (l1_forward nth) shmem._1;

    (* Convert the tensor liveness into a [live_c_shmem] so the post is sendable. *)
    tensor_concr (from_array (l1_forward nth) shmem._1);
    rewrite each core (from_array (l1_forward nth) shmem._1) as shmem._1;
    with vgsa. assert shmem._1 |-> vgsa;
    fold_live_c_shmem shmem._1;

    if_intro_true' (op_Equals #nat tid 0) (
      live_c_shmem #(SHArray et nth) shmem._1 **
      exists* (v : et).
        Cell output (((SZ.v bid <: natlt rows), ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))
    );
    fold kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid tid;
  } else {
    if_elim_false' (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial pre_map_r (ematrix_row vr bid) nth));
    if_elim_false' (op_Equals #nat tid 0) (Cell output (((SZ.v bid <: natlt rows), ()) <: abs (rows @| INil)) |-> (acc1 sout bid));
    if_intro_false' (op_Equals #nat tid 0) (
      live_c_shmem #(SHArray et nth) shmem._1 **
      exists* (v : et).
        Cell output (((SZ.v bid <: natlt rows), ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))
    );
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    fold kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid tid;
    ()
  };
}
#pop-options

(* ── Block-level setup/teardown ────────────────────────────────────────── *)

ghost
fn block_setup_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid))
  ensures
    (forall+ (i : natlt nth). kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  (* share input fractional permission across nth threads *)
  tensor_share_n x nth;

  (* tid 0 gets the output cell *)
  forevery_if_intro #(natlt nth) 0 (fun _ -> Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid));
  forevery_ext
    (fun tid -> if_ (op_Equals #(natlt nth) tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)))
    (fun tid -> if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)));

  forevery_zip (fun _ -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx) _;

  (* View shmem array as a tensor and explode it into per-cell ownership. *)
  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
    (fun tid -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
                if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
       if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid))) **
      Cell (from_array (l1_forward nth) gsa) (abs_bij.gg (tid <: natlt nth))
        |-> (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
    )
    (fun (tid : natlt nth) -> kpre_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };
  ()
}

ghost
fn block_teardown_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et rows)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth). kpost_block pre_map pre_map_r rows cols nth x output sx vr sout shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     exists* (v : et).
       Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
       pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))))
{
  forevery_unzip _ _;

  tensor_gather_n x nth;

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equals #nat tid 0) (
        live_c_shmem #(SHArray et nth) shmem._1 **
        exists* (v : et).
          Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
          pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))))
    (fun tid ->
      if_ (op_Equals #(natlt nth) tid 0) (
        live_c_shmem #(SHArray et nth) shmem._1 **
        exists* (v : et).
          Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
          pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live_c_shmem #(SHArray et nth) shmem._1 **
      exists* (v : et).
        Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))
  );

  fold_live_c_shmems_nil shmem._2 #_;
  fold_live_c_shmems_cons shmem #_;
}

(* ── Outer setup/teardown: share x across blocks, explode output ─────── *)

ghost
fn setup_block_outer
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et rows)
  ()
  norewrite
  requires
    x |-> sx ** output |-> sout
  ensures
    (forall+ (bid : natlt rows).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_pts_to_ref output;
  tensor_share_n x rows;
  tensor_explode output;
  forevery_iso abs_bij _;

  forevery_ext
    (fun (bid : natlt rows) -> Cell output (abs_bij.gg (bid <: natlt rows)) |-> acc sout (abs_bij.gg (bid <: natlt rows)))
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid));

  forevery_zip #(natlt rows)
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid));
  ()
}

#push-options "--z3rlimit 100 --fuel 4 --ifuel 4"
ghost
fn teardown_block_outer
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et rows)
  ()
  norewrite
  requires
    (forall+ (bid : natlt rows).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       exists* (v : et).
         Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
         pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    exists* (sout' : chest1 et rows).
      x |-> sx ** output |-> sout' **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ rsum (lseq_map pre_map_r (ematrix_row vr r)))
{
  forevery_unzip
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt rows) ->
       exists* (v : et).
         Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
         pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))));

  tensor_gather_n x rows;

  (* Skolemize the existential: get a function bid -> et naming each cell value *)
  let f =
    forevery_exists
      (fun (bid : natlt rows) (v : et) ->
         Cell output ((bid, ()) <: abs (rows @| INil)) |-> v **
         pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))));

  (* Build a concrete chest carrying the cell values. *)
  let sout' : chest1 et rows =
    hide (mk1 #et #rows (fun (bid : natlt rows) -> f bid));

  (* Extract the per-row pure approximation fact across all bids. *)
  forevery_extract_pure
    (fun (bid : natlt rows) ->
       Cell output ((bid, ()) <: abs (rows @| INil)) |-> f bid **
       pure (f bid %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))))
    (fun (bid : natlt rows) ->
       (acc1 (reveal sout') bid) %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)))
    fn _ {};

  (* Drop the per-cell pure now that we extracted the global fact. *)
  forevery_drop_pure
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (rows @| INil)) |-> f bid)
    (fun (bid : natlt rows) ->
       f bid %~ rsum (lseq_map pre_map_r (ematrix_row vr bid)));

  forevery_ext
    (fun (bid : natlt rows) ->
       Cell output ((bid, ()) <: abs (rows @| INil)) |-> f bid)
    (fun (bid : natlt rows) ->
       Cell output (abs_bij.gg (bid <: natlt rows)) |-> acc (reveal sout') (abs_bij.gg (bid <: natlt rows)));

  forevery_iso_back (abs_bij #rows)
    (fun (i : abs (rows @| INil)) -> Cell output i |-> acc (reveal sout') i);

  tensor_implode output;
  ()
}
#pop-options

(* ── Block-of sendability of the per-thread pre/post ────────────────────── *)

(* Establish [is_block_array shmem._1] from the shmem invariant. *)
let block_array_of_shmem
  (#et:Type0) {| scalar et |} (nth : szp) (sh : c_shmems [SHArray et nth])
  (pf : squash (c_shmems_inv sh))
  : squash (is_block_array sh._1)
  = ()

(* Block-of sendability of a single cell of a shmem (block-array) tensor.
   The generic global-cell send instance does not apply (the shmem tensor is
   not global), so we build the leaf send by hand: [pts_to_cell] unfolds to a
   slice, whose generic send is keyed on [visibility_of], which is [block_of]
   for a block array. *)
let block_cell_send
  (#et:Type0) {| scalar et |} (nth : szp) (sh : c_shmems [SHArray et nth])
  (pf : squash (c_shmems_inv sh))
  (tid : natlt nth) (v : et) (#f : perm)
  : is_send_across block_of (tensor_pts_to_cell (from_array (l1_forward nth) sh._1) #f (tid, ()) v)
  = block_array_of_shmem nth sh pf;
    let gsa : (a:larray et nth { is_block_array a }) = sh._1 in
    tensor_pts_to_cell_eq (from_array (l1_forward nth) gsa) (tid, ()) f v;
    let idx : nat = (l1_forward nth).imap.f (tid, ()) in
    let inst : is_send_across (visibility_of gsa) (pts_to_cell gsa #f idx v) = TC.solve in
    inst

(* Sendability of [kpre_block]: the global input/output pieces resolve by
   [solve]; the shmem cell is sent explicitly via [block_cell_send]. *)
let kpre_block_sendable
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (gx : squash (is_global x))
  (go : squash (is_global output))
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et rows)
  (sh : c_shmems [SHArray et nth])
  (pf : squash (c_shmems_inv sh))
  (bid : natlt rows) (tid : natlt nth)
  : is_send_across block_of (kpre_block pre_map pre_map_r rows cols nth x output sx vr sout sh bid tid)
  = block_array_of_shmem nth sh pf;
    let p3 : is_send_across block_of
               (exists* (v:et). tensor_pts_to_cell (from_array (l1_forward nth) sh._1) (tid, ()) v)
      = is_send_across_exists
          (fun (v:et) -> tensor_pts_to_cell (from_array (l1_forward nth) sh._1) #1.0R (tid, ()) v)
          #(fun (v:et) -> block_cell_send nth sh pf tid v #1.0R)
    in
    is_send_across_star
      (x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx)
      (if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)) **
       (exists* (v : et). tensor_pts_to_cell (from_array (l1_forward nth) sh._1) (tid, ()) v))
      #TC.solve
      #(is_send_across_star
          (if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid)))
          (exists* (v : et). tensor_pts_to_cell (from_array (l1_forward nth) sh._1) (tid, ()) v)
          #TC.solve
          #p3)

(* Sendability of [kpost_block]: the shmem piece is a [live_c_shmem], whose send
   instance is unambiguous, so [solve] discharges the whole thing. *)
let kpost_block_sendable
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (gx : squash (is_global x))
  (go : squash (is_global output))
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et rows)
  (sh : c_shmems [SHArray et nth])
  (pf : squash (c_shmems_inv sh))
  (bid : natlt rows) (tid : natlt nth)
  : is_send_across block_of (kpost_block pre_map pre_map_r rows cols nth x output sx vr sout sh bid tid)
  = block_array_of_shmem nth sh pf;
    TC.solve

(* ── Kernel descriptor ─────────────────────────────────────────────────── *)

inline_for_extraction noextract
let kdesc_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (sx   : chest2 et   rows cols)
  (vr   : chest2 real rows cols { sx %~ vr })
  (sout : chest1 et rows)
  : kernel_desc
      (x |-> sx ** output |-> sout)
      (exists* (sout' : chest1 et rows).
         x |-> sx ** output |-> sout' **
         pure (forall (r : nat). r < SZ.v rows ==>
               (acc1 sout' r) %~ rsum (lseq_map pre_map_r (ematrix_row vr r))))
  = {
    nblk = rows;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (from_array _ shmem._1)
                          (vr_partial pre_map_r (ematrix_row vr bid) nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth)
                          (from_array _ shmem._1)
                          (vr_partial pre_map_r (ematrix_row vr bid) nth)));

    f = kf_block pre_map pre_map_r rows cols nth x output sx vr sout;

    block_pre  = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      Cell (output <: array1 et lout) ((bid, ()) <: abs (rows @| INil)) |-> (acc1 sout bid));
    block_post = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      exists* (v : et).
        Cell (output <: array1 et lout) ((bid, ()) <: abs (rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map pre_map_r (ematrix_row vr bid))));

    setup    = setup_block_outer    pre_map pre_map_r rows cols nth x output sx vr sout;
    teardown = teardown_block_outer pre_map pre_map_r rows cols nth x output sx vr sout;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup_block    pre_map pre_map_r rows cols nth x output sx vr sout;
    block_teardown = block_teardown_block pre_map pre_map_r rows cols nth x output sx vr sout;

    kpre  = kpre_block  pre_map pre_map_r rows cols nth x output sx vr sout;
    kpost = kpost_block pre_map pre_map_r rows cols nth x output sx vr sout;
    frame = pure (SZ.fits (tlayout_ulen lout));

    kpre_sendable       = (fun sh pf bid tid ->
      kpre_block_sendable pre_map pre_map_r rows cols nth x output () () sx vr sout sh pf bid tid);
    kpost_sendable      = (fun sh pf bid tid ->
      kpost_block_sendable pre_map pre_map_r rows cols nth x output () () sx vr sout sh pf bid tid);
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

(* ── Entry point ──────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (#sx   : chest2 et   rows cols)
  (vr    : chest2 real rows cols)
  (#sout : chest1 et rows)
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (sx %~ vr)
  ensures
    exists* (sout' : chest1 et rows).
      on gpu_loc (output |-> sout') **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ rsum (Kuiper.Seq.Common.lseq_map pre_map_r (ematrix_row vr r)))
{
  launch_sync (kdesc_block pre_map pre_map_r rows cols nth x output sx vr sout);
}
