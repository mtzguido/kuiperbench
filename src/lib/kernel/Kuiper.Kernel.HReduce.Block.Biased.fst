module Kuiper.Kernel.HReduce.Block.Biased

friend Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }
open Kuiper.Kernel.HReduce
open Kuiper.EMatrix
// Re-open after Kuiper.Tensor so the seq-level `@!`/`seq![..]`/`@+` notations
// shadow the shape-level ones brought in by Kuiper.Tensor.
open Kuiper.Seq.Common

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
(* The SMT solver sometimes cannot prove that dividing a positive real
   by a positive integer yields a positive result.  This helper
   establishes the fact once so call sites can use it as a hint. *)
private let perm_div_pos (p : perm) (k : pos)
  : Lemma (p /. k >. 0.0R)
          [SMTPat (p /. k)]
  = ()

(* Per-element step lemma for the strided reduction *)
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

private let stride_length_exact (k n stride off : nat)
  : Lemma
      (requires
        stride > 0 /\ off < stride /\
        k <= (n - off + stride - 1) / stride /\
        k * stride + off >= n)
      (ensures k == (n - off + stride - 1) / stride)
  = Math.Lemmas.lemma_div_le (k * stride) (n - off + stride - 1) stride;
    Math.Lemmas.cancel_mul_div k stride

(* Per-thread biased strided sum: identical to [sum_stride_map_2d] from
   [Kuiper.Kernel.HReduce.Block], but takes a bias element that is
   pre-applied to [pre_map] at each step. *)
#push-options "--fuel 4 --ifuel 8 --z3rlimit 200"
inline_for_extraction noextract
fn sum_stride_map_2d_biased
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map_1 : et -> et)
  (pre_map_1_r : real -> real { pre_map_1 %~ pre_map_1_r })
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
    pure (res %~ rsum (seq_stride (lseq_map pre_map_1_r vr_row) stride off))
{
  let mut acc : et = zero;
  let mut idx : sz = off;
  let gidx = galloc #nat 0;

  while (!idx <^ cols)
    invariant
      live acc ** live gidx **
      live idx ** pure (SZ.v !idx == gread gidx * stride + off) **
      pure (gread gidx <= seq_stride_length (lseq_map pre_map_1_r vr_row) stride off /\
            !idx < cols + stride /\
            !acc %~ rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_1_r vr_row) stride off))) **
      emp
    decreases (cols + stride - !idx)
  {
    assert pure (gread gidx < seq_stride_length vr_row stride off);

    let idx_raw : sz = !idx;
    assert pure (SZ.v idx_raw < SZ.v cols);
    let idx_v : szlt cols = idx_raw;
    let v = read_at rows cols x row idx_v;
    let v' = pre_map_1 v;
    (**)assert (pure (v == acc2 sx row idx_v));
    (**)assert (pure (v %~ (vr_row @! SZ.v idx_v)));
    (**)assert (pure (v' %~ (lseq_map pre_map_1_r vr_row @! SZ.v idx_v)));

    a_add !acc v'
      (rsum (seq_take (gread gidx) (seq_stride (lseq_map pre_map_1_r vr_row) stride off)))
      ((lseq_map pre_map_1_r vr_row) @! SZ.v idx_v);
    rsum_seq_stride_step (lseq_map pre_map_1_r vr_row) stride off (gread gidx);

    let vgidx = gread gidx;
    Math.Lemmas.distributivity_add_left vgidx 1 stride;

    acc := !acc `add` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);
    ()
  };

  stride_length_exact (gread gidx) (Seq.length (lseq_map pre_map_1_r vr_row)) stride off;
  assert pure (gread gidx == seq_stride_length (lseq_map pre_map_1_r vr_row) stride off);
  assert pure (seq_take (seq_stride_length (lseq_map pre_map_1_r vr_row) stride off)
                       (seq_stride (lseq_map pre_map_1_r vr_row) stride off)
              == seq_stride (lseq_map pre_map_1_r vr_row) stride off);
  drop_ (gidx |-> _);
  !acc
}
#pop-options

(* Named partial application of the biased pre_map, used in signatures
   AND bodies so the SMT solver can equate them (lambdas are opaque). *)
let prer
  (pre_map_r : real -> real -> real)
  (#m : nat)
  (vbias : chest1 real m)
  (bid : natlt m)
  : real -> real
  = fun c -> pre_map_r c (acc1 vbias bid)

(* ── Per-thread predicates for the per-block kernel ────────────────────── *)

unfold
let kpre_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols)
  (#lbias : layout1 rows)
  (#lout  : layout1 rows)
  (x      : array2 et lin)
  (bias   : array1 et lbias)
  (output : array1 et lout)
  (sx    : chest2 et   rows cols)
  (vr    : chest2 real rows cols)
  (sbias : chest1 et rows)
  (vbias : chest1 real rows)
  (sout  : chest1 et rows)
  (fbias : perm)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    bias |-> Frac ((fbias /. SZ.v rows) /. nth) sbias **
    if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid)) **
    exists* (v : et). tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ()) v

unfold
let kpost_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols)
  (#lbias : layout1 rows)
  (#lout  : layout1 rows)
  (x      : array2 et lin)
  (bias   : array1 et lbias)
  (output : array1 et lout)
  (sx    : chest2 et   rows cols)
  (vr    : chest2 real rows cols)
  (sbias : chest1 et rows)
  (vbias : chest1 real rows)
  (sout  : chest1 et rows)
  (fbias : perm)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  (tid : natlt nth)
  : slprop
  = x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
    bias |-> Frac ((fbias /. SZ.v rows) /. nth) sbias **
    if_ (op_Equals #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))
    )

(* ── Per-thread kernel function ────────────────────────────────────────── *)

(* [min (a+1) n == a+1] when [a+1 <= n].  Proved in a clean, noise-free
   context: under the heavy nonlinear context of [kf_block] the SMT solver
   fails to unfold [Kuiper.Math.min], so we discharge it standalone. *)
let min_succ (a n : nat)
  : Lemma (requires a + 1 <= n) (ensures Kuiper.Math.min (a + 1) n == a + 1)
  = ()

(* [tid <= y < tid+1 ==> tid == y] is trivial integer reasoning, but under
   the heavy context of [kf_block] the SMT solver does not reliably close
   the quantified form; prove it standalone. *)
let tid_singleton_forall (t : nat)
  : Lemma (forall (y:nat). t <= y /\ y < t + 1 ==> t == y)
  = ()

(* Element-wise corollary of sequence approximation.  Under the heavy
   nonlinear context of [kf_block] the SMT solver does not reliably unfold
   [seq_approximates] to extract the per-index approximation fact, so we
   discharge it in a clean standalone context. *)
let eseq_approx_index (#a:Type) {| scalar a, real_like a |} (#n:nat)
  (s : erased (lseq a n)) (r : erased (lseq real n))
  : Lemma (requires s %~ r)
          (ensures (forall (i:nat). i < n ==> ((reveal s) @! i) %~ ((reveal r) @! i)))
  = ()

(* Chest1 analogue of [eseq_approx_index]: from a chest approximation
   [s %~ r] extract the per-cell approximation at index [i]. *)
let echest_approx_index (#a:Type) {| scalar a, real_like a |} (#n:nat)
  (s : chest1 a n) (r : chest1 real n) (i : natlt n)
  : Lemma (requires s %~ r)
          (ensures (acc1 s i) %~ (acc1 r i))
  = ()

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn kf_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols) {| ctlayout lin   |}
  (#lbias : layout1 rows)             {| ctlayout lbias |}
  (#lout  : layout1 rows)             {| ctlayout lout  |}
  (x      : array2 et lin)
  (bias   : array1 et lbias)
  (output : array1 et lout)
  (sx    : chest2 et   rows cols)
  (vr    : chest2 real rows cols { sx %~ vr })
  (sbias : chest1 et rows)
  (vbias : chest1 real rows { sbias %~ vbias })
  (sout  : chest1 et rows)
  (fbias : perm)
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt rows)
  (tid : szlt nth)
  ()
  requires
    gpu **
    pure (c_shmems_inv shmem) **
    kpre_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1)
                       (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth)) **
    B.barrier_state 0
  ensures
    gpu **
    kpost_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias shmem bid tid **
    thread_id nth tid **
    block_id rows bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1)
                       (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  unfold kpre_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias shmem bid tid;

  let (gsa, _) = shmem;
  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  (* Row of vr at bid *)
  let vr_row : erased (lseq real cols) = hide (ematrix_row (reveal vr) bid);
  let pre_map_bid_r : (real -> real) = prer pre_map_r vbias bid;

  (* Bridge from (sx %~ vr) to row-level approximation. *)
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 (vr_row @! j) == acc2 (reveal vr) bid j);
  assert pure (forall (j:nat). j < SZ.v cols ==>
                 acc2 sx bid j %~ (vr_row @! j));

  (* Read bias value and partially apply pre_map *)
  let bias_bid : et = tensor_read bias (cidx1 bid);
  assert pure (bias_bid == acc1 sbias bid);
  echest_approx_index sbias vbias bid;
  assert pure (bias_bid %~ (acc1 vbias bid));
  let pre_map_bid : (et -> et) = (fun (xx:et) -> pre_map xx bias_bid);

  (* Compute partial sum over stride and write to shmem. *)
  let psum : et = sum_stride_map_2d_biased pre_map_bid pre_map_bid_r rows cols x bid nth tid vr_row;
  tensor_write_cell sa (tid, ()) psum;

  (* Set up tree reduction state. *)
  let mut n : szlt 32 = 0sz;

  let psum_chest : chest1 et 1 = mk1 #et #1 (fun _ -> psum);
  slice_singleton sa tid psum psum_chest;

  (**)fold (array1_pts_to_slice_sum sa tid (tid + 1) (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth));
  assert pure (pow2 (SZ.v !n) == 1);
  assert pure (SZ.v tid + 1 <= SZ.v nth);
  min_succ tid nth;
  rewrite (array1_pts_to_slice_sum sa tid (tid + 1) (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth))
       as (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth));
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth));

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth)) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration nth sa (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth) tid !n;
    FStar.Math.Lemmas.pow2_double_mult (SZ.v !n);
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 it) nth) (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth)))
  as
    (if_ (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth)));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  if (tid = 0sz) {
    if_elim_true' (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth));
    if_elim_true' (op_Equals #nat tid 0) (Cell output (((SZ.v bid <: natlt rows), ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid));
    unfold array1_pts_to_slice_sum sa 0 nth (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth);
    (**)strided_sum_is_sum pre_map_bid_r vr_row nth;

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
    if_intro_true' (op_Equals #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output (((SZ.v bid <: natlt rows), ()) <: abs (SZ.v rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))
    );
    fold kpost_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias shmem bid tid;
  } else {
    if_elim_false' (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth));
    if_elim_false' (op_Equals #nat tid 0) (Cell output (((SZ.v bid <: natlt rows), ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid));
    if_intro_false' (op_Equals #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output (((SZ.v bid <: natlt rows), ()) <: abs (SZ.v rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))
    );
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    fold kpost_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias shmem bid tid;
    ()
  };
}
#pop-options

(* ── Block-level setup/teardown ────────────────────────────────────────── *)

ghost
fn block_setup_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols)
  (#lbias : layout1 rows)
  (#lout  : layout1 rows)
  (x      : array2 et lin)
  (bias   : array1 et lbias)
  (output : array1 et lout)
  (sx    : chest2 et   rows cols)
  (vr    : chest2 real rows cols { sx %~ vr })
  (sbias : chest1 et rows)
  (vbias : chest1 real rows { sbias %~ vbias })
  (sout  : chest1 et rows)
  (fbias : perm)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     bias |-> Frac (fbias /. SZ.v rows) sbias **
     Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid))
  ensures
    (forall+ (i : natlt nth). kpre_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias shmem bid i) **
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
  tensor_share_n bias nth;

  (* tid 0 gets the output cell *)
  forevery_if_intro #(natlt nth) 0 (fun _ -> Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid));
  forevery_ext
    (fun tid -> if_ (op_Equals #(natlt nth) tid 0) (Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid)))
    (fun tid -> if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid)));

  forevery_zip
    (fun (_:natlt nth) -> x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx)
    (fun (tid:natlt nth) ->
       if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid)));

  forevery_zip
    (fun (_:natlt nth) -> bias |-> Frac ((fbias /. SZ.v rows) /. nth) sbias)
    (fun (tid:natlt nth) ->
       x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
       if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid)));

  (* View shmem array as Array1. Explode it. *)
  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
    (fun tid ->
      bias |-> Frac ((fbias /. SZ.v rows) /. nth) sbias **
      (x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
       if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid))))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (bias |-> Frac ((fbias /. SZ.v rows) /. nth) sbias **
       (x |-> Frac ((1.0R /. SZ.v rows) /. nth) sx **
        if_ (op_Equals #nat tid 0) (Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid)))) **
      Cell (from_array (l1_forward nth) gsa) (abs_bij.gg (tid <: natlt nth))
        |-> (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
    )
    (fun (tid : natlt nth) -> kpre_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };
  ()
}

ghost
fn block_teardown_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols)
  (#lbias : layout1 rows)
  (#lout  : layout1 rows)
  (x      : array2 et lin)
  (bias   : array1 et lbias)
  (output : array1 et lout)
  (sx    : chest2 et   rows cols)
  (vr    : chest2 real rows cols { sx %~ vr })
  (sbias : chest1 et rows)
  (vbias : chest1 real rows { sbias %~ vbias })
  (sout  : chest1 et rows)
  (fbias : perm)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt rows)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth). kpost_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (x |-> Frac (1.0R /. SZ.v rows) sx **
     bias |-> Frac (fbias /. SZ.v rows) sbias **
     exists* (v : et).
       Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
       pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid))))
{
  forevery_unzip _ _;

  forevery_unzip _ _;

  tensor_gather_n x nth;
  tensor_gather_n bias nth;

  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equals #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et).
          Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
          pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))))
    (fun tid ->
      if_ (op_Equals #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et).
          Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
          pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et).
        Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))
  );

  tensor_concr (from_array (l1_forward nth) shmem._1);
  rewrite each core (from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

(* ── Outer setup/teardown ─────────────────────────────────────────────── *)

ghost
fn setup_block_outer
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols) {| ctlayout lin   |}
  (#lbias : layout1 rows)             {| ctlayout lbias |}
  (#lout  : layout1 rows)             {| ctlayout lout  |}
  (x      : array2 et lin)
  (bias   : array1 et lbias)
  (output : array1 et lout)
  (sx    : chest2 et   rows cols)
  (vr    : chest2 real rows cols { sx %~ vr })
  (sbias : chest1 et rows)
  (vbias : chest1 real rows { sbias %~ vbias })
  (sout  : chest1 et rows)
  (fbias : perm)
  ()
  norewrite
  requires
    x |-> sx ** bias |-> Frac fbias sbias ** output |-> sout
  ensures
    (forall+ (bid : natlt rows).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       bias |-> Frac (fbias /. SZ.v rows) sbias **
       Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid)) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_pts_to_ref output;
  tensor_share_n x rows;
  tensor_share_n bias rows;
  tensor_explode output;
  forevery_iso abs_bij _;

  forevery_ext
    (fun (bid : natlt rows) -> Cell output (abs_bij.gg (bid <: natlt rows)) |-> acc sout (abs_bij.gg (bid <: natlt rows)))
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid));

  forevery_zip #(natlt rows)
    (fun (_ : natlt rows) -> bias |-> Frac (fbias /. SZ.v rows) sbias)
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid));
  forevery_zip #(natlt rows)
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt rows) ->
       bias |-> Frac (fbias /. SZ.v rows) sbias **
       Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid));
  ()
}

#push-options "--z3rlimit 100 --fuel 4 --ifuel 4"
ghost
fn teardown_block_outer
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols) {| ctlayout lin   |}
  (#lbias : layout1 rows)             {| ctlayout lbias |}
  (#lout  : layout1 rows)             {| ctlayout lout  |}
  (x      : array2 et lin)
  (bias   : array1 et lbias)
  (output : array1 et lout)
  (sx    : chest2 et   rows cols)
  (vr    : chest2 real rows cols { sx %~ vr })
  (sbias : chest1 et rows)
  (vbias : chest1 real rows { sbias %~ vbias })
  (sout  : chest1 et rows)
  (fbias : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt rows).
       x |-> Frac (1.0R /. SZ.v rows) sx **
       bias |-> Frac (fbias /. SZ.v rows) sbias **
       exists* (v : et).
         Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
         pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    exists* (sout' : chest1 et rows).
      x |-> sx ** bias |-> Frac fbias sbias ** output |-> sout' **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ rsum (lseq_map (prer pre_map_r vbias r) (ematrix_row vr r)))
{
  forevery_unzip
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (bid : natlt rows) ->
       bias |-> Frac (fbias /. SZ.v rows) sbias **
       (exists* (v : et).
          Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
          pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))));

  forevery_unzip
    (fun (_ : natlt rows) -> bias |-> Frac (fbias /. SZ.v rows) sbias)
    (fun (bid : natlt rows) ->
       exists* (v : et).
         Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
         pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid))));

  tensor_gather_n x rows;
  tensor_gather_n bias rows;

  (* Skolemize the existential *)
  let f =
    forevery_exists
      (fun (bid : natlt rows) (v : et) ->
         Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
         pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid))));

  let sout' : chest1 et rows =
    hide (mk1 #et #rows (fun (bid : natlt rows) -> f bid));

  forevery_extract_pure
    (fun (bid : natlt rows) ->
       Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> f bid **
       pure (f bid %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid))))
    (fun (bid : natlt rows) ->
       (acc1 (reveal sout') bid) %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)))
    fn _ {};

  forevery_drop_pure
    (fun (bid : natlt rows) -> Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> f bid)
    (fun (bid : natlt rows) ->
       f bid %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid)));

  forevery_ext
    (fun (bid : natlt rows) ->
       Cell output ((bid, ()) <: abs (SZ.v rows @| INil)) |-> f bid)
    (fun (bid : natlt rows) ->
       Cell output (abs_bij.gg (bid <: natlt rows)) |-> acc (reveal sout') (abs_bij.gg (bid <: natlt rows)));

  forevery_iso_back (abs_bij #rows)
    (fun (i : abs (SZ.v rows @| INil)) -> Cell output i |-> acc (reveal sout') i);

  tensor_implode output;
  ()
}
#pop-options

(* ── Kernel descriptor ─────────────────────────────────────────────────── *)

inline_for_extraction noextract
let kdesc_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols) {| ctlayout lin   |}
  (#lbias : layout1 rows)             {| ctlayout lbias |}
  (#lout  : layout1 rows)             {| ctlayout lout  |}
  (x      : array2 et lin   { is_global x      })
  (bias   : array1 et lbias { is_global bias   })
  (output : array1 et lout  { is_global output })
  (sx    : chest2 et   rows cols)
  (vr    : chest2 real rows cols { sx %~ vr })
  (sbias : chest1 et rows)
  (vbias : chest1 real rows { sbias %~ vbias })
  (sout  : chest1 et rows)
  (fbias : perm)
  : kernel_desc
      (x |-> sx ** bias |-> Frac fbias sbias ** output |-> sout)
      (exists* (sout' : chest1 et rows).
         x |-> sx ** bias |-> Frac fbias sbias ** output |-> sout' **
         pure (forall (r : nat). r < SZ.v rows ==>
               (acc1 sout' r) %~ rsum (lseq_map (prer pre_map_r vbias r) (ematrix_row vr r))))
  = {
    nblk = rows;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (from_array _ shmem._1)
                          (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth)
                          (from_array _ shmem._1)
                          (vr_partial (prer pre_map_r vbias bid) (ematrix_row vr bid) nth)));

    f = kf_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias;

    block_pre  = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      bias |-> Frac (fbias /. SZ.v rows) sbias **
      Cell (output <: array1 et lout) ((bid, ()) <: abs (SZ.v rows @| INil)) |-> (acc1 sout bid));
    block_post = (fun bid ->
      x |-> Frac (1.0R /. SZ.v rows) sx **
      bias |-> Frac (fbias /. SZ.v rows) sbias **
      exists* (v : et).
        Cell (output <: array1 et lout) ((bid, ()) <: abs (SZ.v rows @| INil)) |-> v **
        pure (v %~ rsum (lseq_map (prer pre_map_r vbias bid) (ematrix_row vr bid))));

    setup    = setup_block_outer    pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias;
    teardown = teardown_block_outer pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup_block    pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias;
    block_teardown = block_teardown_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias;

    kpre  = kpre_block  pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias;
    kpost = kpost_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias;
    frame = pure (SZ.fits (tlayout_ulen lout));

    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

(* ── Entry point ──────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_block_biased
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 rows cols) {| ctlayout lin   |}
  (#lbias : layout1 rows)             {| ctlayout lbias |}
  (#lout  : layout1 rows)             {| ctlayout lout  |}
  (x      : array2 et lin   { is_global x      })
  (bias   : array1 et lbias { is_global bias   })
  (output : array1 et lout  { is_global output })
  (#sx    : chest2 et   rows cols)
  (vr     : chest2 real rows cols)
  (#sbias : chest1 et rows)
  (vbias  : chest1 real rows)
  (#sout  : chest1 et rows)
  (#fbias : perm)
  preserves
    cpu **
    on gpu_loc (x |-> sx) **
    on gpu_loc (bias |-> Frac fbias sbias)
  requires
    on gpu_loc (output |-> sout) **
    pure (sx %~ vr) **
    pure (sbias %~ vbias)
  ensures
    exists* (sout' : chest1 et rows).
      on gpu_loc (output |-> sout') **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ rsum (Kuiper.Seq.Common.lseq_map (prer pre_map_r vbias r)
                                                             (ematrix_row vr r)))
{
  launch_sync (kdesc_block pre_map pre_map_r rows cols nth x bias output sx vr sbias vbias sout fbias);
}
