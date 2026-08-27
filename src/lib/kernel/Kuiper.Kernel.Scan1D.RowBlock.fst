module Kuiper.Kernel.Scan1D.RowBlock

(* Pulse implementation of the row-per-block inclusive prefix-scan
 * declared in the .fsti.  See the .fsti for the strategic comment.
 *
 * Structure:
 *   - [kpre] / [kpost]: per-block precondition / postcondition.
 *     Each block owns [1 / SZ.v rows] of [input] and the full row
 *     [bid] of [output] as a [forall+ (c : natlt cols)] of cells.
 *   - [kf]: per-block kernel.  Single-thread sequential scan over
 *     [input[bid, 0 .. cols)], extracting one output cell at a
 *     time from the forall+, writing it, and re-inserting via the
 *     trade produced by [forevery_extract'].
 *   - [setup] / [teardown]: distribute / collect the per-row
 *     resources to / from the block-wise forall+.
 *   - [kdesc]: assembles into a [kernel_desc_m_1] (one block per
 *     row, one thread per block, no shmem, no barrier).
 *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Scan1D
open Kuiper.Monoid.Reduce
open Kuiper.Seq.Common
open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module Seq = FStar.Seq

(* ──────────────────────────────────────────────────────────────────────
 * Section 1: prefix-fold lemmas (local copy of those in
 * [Kuiper.Kernel.Scan1D]; redefined here rather than [friend]ed to
 * keep verification context small).
 * ────────────────────────────────────────────────────────────────────── *)

let scan_partial
  (#t : Type0) (m : cmonoid t)
  (s : Seq.seq t) (di : nat{di <= Seq.length s})
  : GTot t
  = red_fold m m.rid (Seq.slice s 0 di)

let scan_partial_zero
  (#t : Type0) (m : cmonoid t)
  (s : Seq.seq t)
  : Lemma (scan_partial m s 0 == m.rid)
  = let s0 = Seq.slice s 0 0 in
    Seq.lemma_eq_intro s0 Seq.empty;
    ()

#push-options "--z3rlimit 30"
let scan_partial_step
  (#t : Type0) (m : cmonoid t)
  (s : Seq.seq t) (di : nat{di < Seq.length s})
  : Lemma (scan_partial m s (di + 1)
           == m.rop (scan_partial m s di) (Seq.index s di))
  = let pre  : Seq.seq t = Seq.slice s 0 di in
    let pre' : Seq.seq t = Seq.slice s 0 (di + 1) in
    let one  : Seq.seq t = Seq.slice s di (di + 1) in
    Seq.lemma_eq_intro pre' (Seq.append pre one);
    Seq.lemma_eq_intro one (Seq.create 1 (Seq.index s di));
    red_fold_append m pre one;
    ()
#pop-options

let scan_partial_eq_inclusive_at
  (#t : Type0) (m : cmonoid t)
  (s : Seq.seq t) (j : nat{j < Seq.length s})
  : Lemma (scan_partial m s (j + 1) == scan_inclusive_at m s j)
  = ()

let macc_scan2d_inclusive_result
  (#t : Type0)
  (m : cmonoid t)
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (r : natlt rows) (j : natlt cols)
  : Lemma (acc2 (scan2d_inclusive_result m sx) r j ==
           scan_inclusive_at m (ematrix_row sx r) j)
  = ()

(* Local helper lemma used post-loop in [kf]: after the loop, [di_v = cols],
 * so [if c < di_v then ... else ...] always takes the [then] branch
 * for [c : natlt cols]. *)
let scan_postloop_eq
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (#rows #cols : nat)
  (sx   : chest2 t rows cols)
  (sout : chest2 t rows cols)
  (row_view : Seq.lseq t cols)
  (bid : natlt rows)
  (di_v : nat { di_v == cols /\ row_view == ematrix_row sx bid })
  (c : natlt cols)
  : Lemma
      ((if c < di_v
          then scan_inclusive_at m row_view c
          else acc2 sout bid c)
       == scan_inclusive_at m (ematrix_row sx bid) c)
  = ()

(* ──────────────────────────────────────────────────────────────────────
 * Section 2: kpre / kpost predicates
 *
 * Per-block: own [1 / rows] of [input] plus a [forall+ c] of the
 * row [bid] of [output]'s cells.  In [kpre] each cell holds the
 * initial [acc2 sout bid c]; in [kpost] each holds
 * [scan_inclusive_at m (row sx bid) c].
 * ────────────────────────────────────────────────────────────────────── *)

unfold
let kpre
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (rows : szp)
  (cols : nat)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  (input  : array2 t lin)
  (output : array2 t lout)
  (sx   : chest2 t rows cols)
  (sout : chest2 t rows cols)
  (fIn : perm)
  (bid : natlt rows)
  : slprop
  = input |-> Frac (fIn /. SZ.v rows) sx **
    (forall+ (c : natlt cols).
       tensor_pts_to_cell output (idx2 bid c) (acc2 sout bid c))

unfold
let kpost
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (rows : szp)
  (cols : nat)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  (input  : array2 t lin)
  (output : array2 t lout)
  (sx   : chest2 t rows cols)
  (fIn : perm)
  (bid : natlt rows)
  : slprop
  = input |-> Frac (fIn /. SZ.v rows) sx **
    (forall+ (c : natlt cols).
       tensor_pts_to_cell output (idx2 bid c)
         (scan_inclusive_at m (ematrix_row sx bid) c))

(* ──────────────────────────────────────────────────────────────────────
 * Section 3: per-block kernel function (sequential row scan)
 * ────────────────────────────────────────────────────────────────────── *)

(* Loop-invariant cell predicate at iteration [di_v]: cells [c < di_v]
 * already hold the scanned value, cells [c >= di_v] still hold the
 * initial [acc2 sout bid c]. *)
unfold
let cell_at_iter
  (#t : Type0) (m : cmonoid t)
  (#rows #cols : nat)
  (#lout : layout2 rows cols)
  (output : array2 t lout)
  (sx   : chest2 t rows cols)
  (sout : chest2 t rows cols)
  (bid : natlt rows)
  (di_v : nat)
  (c : natlt cols)
  : slprop
  = tensor_pts_to_cell output (idx2 bid c)
      (if c < di_v
         then scan_inclusive_at m (ematrix_row sx bid) c
         else acc2 sout bid c)

#push-options "--z3rlimit 200 --fuel 2 --ifuel 2"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  {| _ : ctlayout lin, _ : ctlayout lout |}
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et rows cols)
  (#sout : chest2 et rows cols)
  (#fIn : perm)
  (bid : szlt rows)
  ()
  preserves gpu
  requires
    kpre m rows cols input output sx sout fIn bid **
    block_id rows bid
  ensures
    kpost m rows cols input output sx fIn bid **
    block_id rows bid
{
  unfold kpre m rows cols input output sx sout fIn bid;

  let row_view : erased (Seq.lseq et cols) = ematrix_row sx bid;

  scan_partial_zero m (reveal row_view);

  (* Fold the cell predicate into the [cell_at_iter] form so the
   * loop invariant can refer to it uniformly. *)
  Kuiper.ForEvery.forevery_ext _
    (fun (c : natlt cols) ->
       cell_at_iter m output sx sout bid 0 c);

  let mut acc : et = m.rid;
  let mut di_ref : SZ.t = 0sz;

  while (!di_ref <^ cols)
    invariant exists* (di_v : SZ.t) (acc_v : et).
      di_ref |-> di_v **
      acc |-> acc_v **
      input |-> Frac (fIn /. SZ.v rows) sx **
      (forall+ (c : natlt cols).
         cell_at_iter m output sx sout bid di_v c) **
      pure (SZ.v di_v <= SZ.v cols /\
            acc_v == scan_partial m (reveal row_view) di_v)
    decreases (SZ.v cols - SZ.v !di_ref)
  {
    let di_old : SZ.t = !di_ref;
    let di_old_sz : szlt cols = di_old;

    (* Read input cell [(bid, di_old)]. *)
    let v : et = tensor_read input (cidx2 (bid <: szlt rows) (di_old_sz <: szlt cols));
    assert pure (v == acc2 sx bid di_old_sz);
    assert pure (Seq.index (reveal row_view) di_old_sz
                 == acc2 sx bid di_old_sz);

    (* Extract the cell at [di_old] from the [forall+], obtaining
     * the cell predicate plus a trade to restore the forall+ with
     * a (possibly different) per-cell slprop. *)
    Kuiper.ForEvery.forevery_extract' #(natlt cols) di_old_sz
      (cell_at_iter m output sx sout bid di_old);

    (* At [c = di_old] the [if c < di_v] branch is false, so we
     * own the cell at value [acc2 sout bid di_old]. *)
    rewrite (cell_at_iter m output sx sout bid di_old di_old_sz)
         as (tensor_pts_to_cell output (idx2 (SZ.v bid <: natlt rows)
                                        (SZ.v di_old_sz <: natlt cols))
              (acc2 sout bid di_old_sz));

    (* Update the running accumulator. *)
    scan_partial_step m (reveal row_view) di_old_sz;
    let acc_old : et = !acc;
    acc := m.rop acc_old v;
    let acc_new : et = !acc;
    assert pure (acc_new == scan_partial m (reveal row_view) (SZ.v di_old_sz + 1));
    scan_partial_eq_inclusive_at m (reveal row_view) di_old_sz;
    assert pure (acc_new == scan_inclusive_at m (reveal row_view) di_old_sz);

    (* Write the new value into the output cell. *)
    tensor_write_cell output (cidx2 (bid <: szlt rows) (di_old_sz <: szlt cols)) acc_new;

    let di_new : SZ.t = di_old +^ 1sz;

    (* Re-fold the cell into the [cell_at_iter] form at the new
     * [di_v = di_new = di_old + 1].  At [c = di_old] the branch is now
     * [if c < di_new] = true, so the body is
     * [scan_inclusive_at m row_view di_old], matching [acc_new]. *)
    rewrite (tensor_pts_to_cell output (idx2 (SZ.v bid <: natlt rows)
                                        (SZ.v di_old_sz <: natlt cols))
               acc_new)
         as (cell_at_iter m output sx sout bid di_new
              di_old_sz);

    (* Instantiate the trade's [forall*] slprop with the new
     * iteration's cell predicate.  Pointwise equality at
     * [c =!= di_old] holds because [c < di_old] ↔ [c < di_new]
     * whenever [c =!= di_old]. *)
    Pulse.Lib.Forall.elim_forall
      (cell_at_iter m output sx sout bid di_new);
    Pulse.Lib.Trade.elim_trade _ _;

    (* Canonical increment form — must be the last statement in the
     * loop body so Karamel emits [di_ref++] instead of hoisting
     * [di_ref = di_old + 1] (which references a body-local). *)
    di_ref := SZ.( !di_ref +^ 1sz );
    with new_di_v. assert (di_ref |-> new_di_v);
    assert pure (SZ.v new_di_v == SZ.v di_new);
    rewrite each new_di_v as di_new;
    ()
  };

  (* After the loop, [di_v = cols], so every cell holds
   * [scan_inclusive_at m row_view c]. *)
  with di_v _acc_v. assert
    (di_ref |-> di_v ** acc |-> _acc_v **
     pure (SZ.v di_v == SZ.v cols));

  let bid_n : Ghost.erased (natlt rows) = SZ.v bid;
  rewrite each (SZ.v bid) as (Ghost.reveal bid_n);

  FStar.Classical.forall_intro
    (scan_postloop_eq m (reveal sx) (reveal sout) (reveal row_view)
       (Ghost.reveal bid_n) di_v);

  Kuiper.ForEvery.forevery_ext _
    (fun (c : natlt cols) ->
       tensor_pts_to_cell output (idx2 (Ghost.reveal bid_n) c)
         (scan_inclusive_at m (ematrix_row sx (Ghost.reveal bid_n)) c));

  rewrite each (Ghost.reveal bid_n) as (SZ.v bid);

  fold (kpost m rows cols input output sx fIn bid);
  ()
}
#pop-options

(* ──────────────────────────────────────────────────────────────────────
 * Section 4: outer setup / teardown
 *
 * setup:  input |-> Frac fIn sx ** output |-> sout
 *      → forall+ (bid : natlt rows). kpre bid
 *
 * teardown is symmetric.
 * ────────────────────────────────────────────────────────────────────── *)

#push-options "--z3rlimit 60"
ghost
fn setup_rowblock
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et rows cols)
  (#sout : chest2 et rows cols)
  (#fIn  : perm)
  ()
  norewrite
  requires
    input |-> Frac fIn sx ** output |-> sout
  ensures
    (forall+ (bid : natlt rows).
       kpre m rows cols input output sx sout fIn bid) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_share_n input rows;
  tensor_ilower2 output;

  (* Now: forall+ (_ : natlt rows). input |-> Frac (fIn /. rows) sx
   *      forall+ (r : natlt rows) (c : natlt cols). pts_to_cell output (r, c) (acc2 sout r c)
   * The latter, in Pulse's [forall+ x y. ...] syntax, is curried, so
   * it is exactly [forall+ r. (forall+ c. pts_to_cell output (r, c) (acc2 sout r c))]. *)
  Kuiper.ForEvery.forevery_zip #(natlt rows)
    (fun (_ : natlt rows) ->
       input |-> Frac (fIn /. SZ.v rows) sx)
    (fun (bid : natlt rows) ->
       forall+ (c : natlt cols).
         tensor_pts_to_cell output (idx2 bid c) (acc2 sout bid c));

  Kuiper.ForEvery.forevery_ext _
    (fun (bid : natlt rows) ->
       kpre m rows cols input output sx sout fIn bid);
  ()
}
#pop-options

#push-options "--z3rlimit 60"
ghost
fn teardown_rowblock
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et rows cols)
  (#fIn  : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt rows).
       kpost m rows cols input output sx fIn bid) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    input |-> Frac fIn sx **
    output |-> scan2d_inclusive_result m sx
{
  Kuiper.ForEvery.forevery_ext _
    (fun (bid : natlt rows) ->
       (input |-> Frac (fIn /. SZ.v rows) sx) **
       (forall+ (c : natlt cols).
          tensor_pts_to_cell output (idx2 bid c)
            (scan_inclusive_at m (ematrix_row sx bid) c)));

  Kuiper.ForEvery.forevery_unzip #(natlt rows)
    (fun (_ : natlt rows) ->
       input |-> Frac (fIn /. SZ.v rows) sx)
    (fun (bid : natlt rows) ->
       forall+ (c : natlt cols).
         tensor_pts_to_cell output (idx2 bid c)
           (scan_inclusive_at m (ematrix_row sx bid) c));

  tensor_gather_n input rows;

  (* Rewrite per-cell value from [scan_inclusive_at] to [acc2
   * (scan2d_inclusive_result m sx)] (definitionally equal). *)
  Classical.forall_intro_2
    (macc_scan2d_inclusive_result #et m #rows #cols sx);
  Kuiper.ForEvery.forevery_ext_2
    (fun (bid : natlt rows) (c : natlt cols) ->
       tensor_pts_to_cell output (idx2 bid c)
         (scan_inclusive_at m (ematrix_row sx bid) c))
    (fun (bid : natlt rows) (c : natlt cols) ->
       tensor_pts_to_cell output (idx2 bid c)
         (acc2 (scan2d_inclusive_result m sx) bid c));

  tensor_iraise2 output;
  ()
}
#pop-options

(* ──────────────────────────────────────────────────────────────────────
 * Section 5: kernel descriptor
 * ────────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  {| _ : ctlayout lin, _ : ctlayout lout |}
  (input  : array2 et lin  { is_global input  })
  (output : array2 et lout { is_global output })
  (#sx   : chest2 et rows cols)
  (#sout : chest2 et rows cols)
  (#fIn  : perm)
  : kernel_desc
      (requires
        input  |-> Frac fIn sx **
        output |-> sout)
      (ensures
        input  |-> Frac fIn sx **
        output |-> scan2d_inclusive_result m sx)
  = {
    nblk = rows;
    f = (fun (bid : szlt rows) ->
           kf m rows cols input output #sx #sout #fIn bid);
    frame = pure (SZ.fits (tlayout_ulen lout));
    setup    = setup_rowblock    m rows cols input output #sx #sout #fIn;
    teardown = teardown_rowblock m rows cols input output #sx #fIn;
    kpre  = (fun (bid : natlt rows) ->
              kpre m rows cols input output sx sout fIn bid);
    kpost = (fun (bid : natlt rows) ->
              kpost m rows cols input output sx fIn bid);
    kpre_sendable  = solve;
    kpost_sendable = solve;
  } <: kernel_desc_m_1 _ _

(* ──────────────────────────────────────────────────────────────────────
 * Section 6: launcher
 * ────────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn scan1d_inclusive_rowblock
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout2 rows cols) {| ctlayout lout |}
  (input  : array2 et lin  { is_global input  })
  (output : array2 et lout { is_global output })
  (#sx   : chest2 et rows cols)
  (#sout : chest2 et rows cols)
  (#fIn  : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    on gpu_loc (output |-> sout)
  ensures
    on gpu_loc (output |-> scan2d_inclusive_result m sx)
{
  launch_sync (kdesc m rows cols input output)
}
