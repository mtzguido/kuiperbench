module Kuiper.Kernel.Scan1D

(* Pulse implementation of the polymorphic 2-D row-batched inclusive
 * prefix-scan primitive declared in [Kuiper.Kernel.Scan1D.fsti].
 *
 * For each output cell [(r, j)] the kernel computes
 *
 *   out[r, j] = m.rop-fold of input[r, 0..j+1]
 *             = scan_inclusive_at m (ematrix_row sx r) j
 *
 * One thread per output cell (gid = r * cols + j); thread reads
 * input[r, 0], input[r, 1], ..., input[r, j] sequentially and folds
 * with [m.rop] starting from [m.rid].
 *
 * The kernel is polymorphic over [cmonoid t] so a single Pulse proof
 * covers both cumulative-sum ([cmonoid_fadd_f32]) and cumulative-
 * product ([cmonoid_fmul_f32]) instantiations.  The bridge from this
 * fold-form postcondition to the [Kuiper.Spec.Scan1D]
 * [scan2d_inclusive_post] / variants predicates is delivered by per-
 * challenge wrappers (Kuiper.KB.{CumSum,CumProd,...}).
 *
 * The structure follows [Kuiper.Kernel.WindowReduce1D] line-for-line:
 * each thread owns one output cell + 1/(rows*cols) of the input, runs
 * a sequential loop reading input cells and folding into [acc], then
 * writes [acc] to its output cell.  Setup/teardown bridge the
 * post-[share_n] / post-[ilower] forall+ quantifier into the [kpre]
 * forall+ form using the same [forevery_*] combinator recipe as
 * [WindowReduce1D] (see [setup_fold_kpre_step] / [teardown_unfold_kpost_step]).
 * No arithmetic obligations are admitted. *)

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
 * Section 1: prefix-fold lemmas for the per-thread invariant
 * ────────────────────────────────────────────────────────────────────── *)

(* Inclusive prefix at length [di]: fold of [s[0..di]] under [m].
 * For [di == 0] this is [m.rid]; for [di == i+1] this equals
 * [scan_inclusive_at m s i].  The per-thread loop maintains
 * [acc == scan_partial m s di] at iteration [di]. *)
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
    (* red_fold m m.rid one == m.rop m.rid s[di] == s[di] by neutrality *)
    ()
#pop-options

(* After looping di = 0..j+1, [acc == scan_partial m s (j+1)],
 * which equals [scan_inclusive_at m s j] by definition. *)
let scan_partial_eq_inclusive_at
  (#t : Type0) (m : cmonoid t)
  (s : Seq.seq t) (j : nat{j < Seq.length s})
  : Lemma (scan_partial m s (j + 1) == scan_inclusive_at m s j)
  = ()

(* Per-cell acc2 equality for [scan2d_inclusive_result].  Mathematically
 * direct from [mk2]'s defining equation: at index [(r, j)] the body
 * [scan_inclusive_at m (ematrix_row sx r) j] is exactly the value. *)
let macc_scan2d_inclusive_result
  (#t : Type0)
  (m : cmonoid t)
  (#rows #cols : nat)
  (sx : chest2 t rows cols)
  (r : natlt rows) (j : natlt cols)
  : Lemma (acc2 (scan2d_inclusive_result m sx) r j ==
           scan_inclusive_at m (ematrix_row sx r) j)
  = ()

(* ──────────────────────────────────────────────────────────────────────
 * Section 2: kpre / kpost predicates
 * ────────────────────────────────────────────────────────────────────── *)

unfold
let kpre
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (#rows : nat) (#cols : nat)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  (input  : array2 t lin)
  (output : array2 t lout)
  (sx   : chest2 t rows cols)
  (sout : chest2 t rows cols)
  (fIn : perm)
  (gid : natlt (rows * cols))
  : slprop
  = input |-> Frac (fIn /. (rows * cols)) sx **
    tensor_pts_to_cell output (idx2 (gid / cols) (gid % cols))
      (acc2 sout (gid / cols) (gid % cols))

unfold
let kpost
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (#rows : nat) (#cols : nat)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  (input  : array2 t lin)
  (output : array2 t lout)
  (sx   : chest2 t rows cols)
  (fIn : perm)
  (gid : natlt (rows * cols))
  : slprop
  = input |-> Frac (fIn /. (rows * cols)) sx **
    tensor_pts_to_cell output (idx2 (gid / cols) (gid % cols))
      (scan_inclusive_at m (ematrix_row sx (gid / cols)) (gid % cols))

(* ──────────────────────────────────────────────────────────────────────
 * Section 3: per-thread function (sequential prefix scan)
 * ────────────────────────────────────────────────────────────────────── *)

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (#rows : SZ.t) (#cols : SZ.t)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout2 (SZ.v rows) (SZ.v cols))
  {| _ : ctlayout lin, _ : ctlayout lout |}
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et (SZ.v rows) (SZ.v cols))
  (#sout : chest2 et (SZ.v rows) (SZ.v cols))
  (#fIn : perm)
  (gid : szlt (rows *^ cols))
  ()
  requires
    gpu **
    kpre m input output sx sout fIn (SZ.v gid)
  ensures
    gpu **
    kpost m input output sx fIn (SZ.v gid)
{
  let r_sz : szlt rows = gid /^ cols;
  let j_sz : szlt cols = gid %^ cols;
  assert (pure (SZ.v r_sz == SZ.v gid / SZ.v cols /\ SZ.v j_sz == SZ.v gid % SZ.v cols));
  rewrite each (SZ.v gid / SZ.v cols) as (SZ.v r_sz);
  rewrite each (SZ.v gid % SZ.v cols) as (SZ.v j_sz);

  let row_view : erased (Seq.lseq et (SZ.v cols)) = ematrix_row sx (SZ.v r_sz);

  scan_partial_zero m (reveal row_view);

  let mut acc : et = m.rid;
  let mut di_ref : SZ.t = 0sz;

  let bound : SZ.t = j_sz +^ 1sz;

  while (!di_ref <^ bound)
    invariant exists* (di_v : SZ.t) (acc_v : et).
      di_ref |-> di_v **
      acc |-> acc_v **
      input |-> Frac (fIn /. (SZ.v (rows *^ cols))) sx **
      tensor_pts_to_cell output (idx2 (SZ.v r_sz <: natlt (SZ.v rows)) (SZ.v j_sz <: natlt (SZ.v cols)))
        (acc2 sout (SZ.v r_sz) (SZ.v j_sz)) **
      pure (SZ.v di_v <= SZ.v bound /\
            acc_v == scan_partial m (reveal row_view) (SZ.v di_v))
    decreases (SZ.v bound - SZ.v !di_ref)
  {
    let di_v_raw : SZ.t = !di_ref;
    let di_v : szlt cols = di_v_raw;
    let v : et = tensor_read input (cidx2 (r_sz <: szlt (SZ.v rows)) (di_v <: szlt (SZ.v cols)));

    assert (pure (v == acc2 sx (SZ.v r_sz) (SZ.v di_v)));
    assert (pure (Seq.index (reveal row_view) (SZ.v di_v) == acc2 sx (SZ.v r_sz) (SZ.v di_v)));

    scan_partial_step m (reveal row_view) (SZ.v di_v);
    acc := m.rop !acc v;
    di_ref := !di_ref +^ 1sz
  };

  scan_partial_eq_inclusive_at m (reveal row_view) (SZ.v j_sz);
  let acc_final : et = !acc;
  assert pure (acc_final == scan_inclusive_at m (reveal row_view) (SZ.v j_sz));
  assert pure (reveal row_view == ematrix_row sx (SZ.v r_sz));
  tensor_write_cell output (cidx2 (r_sz <: szlt (SZ.v rows)) (j_sz <: szlt (SZ.v cols))) acc_final;
  rewrite each (SZ.v r_sz) as (SZ.v gid / SZ.v cols);
  rewrite each (SZ.v j_sz) as (SZ.v gid % SZ.v cols);
  ()
}
#pop-options

(* ──────────────────────────────────────────────────────────────────────
 * Section 4: setup / teardown / kdesc
 *
 * Identical structure to [Kuiper.Kernel.WindowReduce1D]: setup performs
 * [share_n] on input + [ilower] on output, then a single named helper
 * ([setup_fold_kpre_step]) bridges the resulting forall+'s into the
 * [kpre] forall+ form using [forevery_unfactor'] / [forevery_zip] /
 * [forevery_ext].  Teardown is symmetric.  The kernel_desc
 * setup/teardown fields plug these helpers in via the thin
 * [kdesc_setup] / [kdesc_teardown] bridges, which use [forevery_map']
 * to rebind the [forall+] from [natlt (SZ.v rows * SZ.v cols)] (helper
 * form) to [natlt (SZ.v (rows *^ cols))] (the [kernel_desc_n] form);
 * the two index bounds are provably equal.
 * ────────────────────────────────────────────────────────────────────── *)

(* Local helpers (mirror of [Kuiper.Kernel.WindowReduce1D]'s opaque
 * index-pair builders) used to keep Pulse's slprop matcher from
 * prematurely unifying the alpha-renamed [forall+] binders. *)
let tid_to_rc (rows cols : nat) (tid : natlt (rows * cols)) : abs (rows @| cols @| INil) =
  idx2 (tid / cols) (tid % cols)

let ait_zero (rows cols : nat) (_ : squash (rows >= 1 /\ cols >= 1))
  : abs (rows @| cols @| INil)
  = idx2 (0 <: natlt rows) (0 <: natlt cols)

(* Helper: a single [pts_to_cell] implies that the cell array's
 * underlying GPU array length [SZ.fits]. *)
ghost
fn pts_to_cell_fits
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l)
  (ij : abs (rows @| cols @| INil))
  (#f : perm) (#v : et)
  preserves tensor_pts_to_cell a #f ij v
  ensures pure (SZ.fits (tlayout_ulen l))
{
  let i : nat = l.imap.f ij;
  tensor_pts_to_cell_eq a ij f v;
  rewrite tensor_pts_to_cell a #f ij v
       as pts_to_cell (core a) #f i v;
  rewrite pts_to_cell (core a) #f i v
       as pts_to_slice (core a) #f i (i + 1) (Seq.cons v Seq.empty);
  pts_to_slice_ref (core a) i (i + 1);
  rewrite pts_to_slice (core a) #f i (i + 1) (Seq.cons v Seq.empty)
       as pts_to_cell (core a) #f i v;
  rewrite pts_to_cell (core a) #f i v
       as tensor_pts_to_cell a #f ij v;
}

(* Setup leaf: fold the post-[share_n] frac quantifier and post-[ilower]
 * cell quantifier into the kpre forall+. *)
ghost
fn setup_fold_kpre_step
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (#rows : nat) (#cols : nat)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx    : chest2 et rows cols)
  (#sout  : chest2 et rows cols)
  (#fIn   : perm)
  ()
  norewrite
  requires
    (forall+ (_ : natlt (rows * cols)).
       input |-> Frac (fIn /. (rows * cols)) sx) **
    (forall+ (r : natlt rows) (c : natlt cols).
       tensor_pts_to_cell output (idx2 r c) (acc2 sout r c))
  ensures
    forall+ (tid : natlt (rows * cols)).
      kpre m input output sx sout fIn tid
{
  forevery_unfactor' (rows * cols) rows cols
    (fun (r : natlt rows) (c : natlt cols) ->
       tensor_pts_to_cell output (idx2 r c) (acc2 sout r c));
  forevery_zip #(natlt (rows * cols))
    (fun (_ : natlt (rows * cols)) -> input |-> Frac (fIn /. (rows * cols)) sx)
    (fun (tid : natlt (rows * cols)) ->
       tensor_pts_to_cell output (tid_to_rc rows cols tid)
         (acc2 sout (tid / cols) (tid % cols)));
  forevery_ext _ (kpre m input output sx sout fIn);
  ()
}

(* Teardown leaf: unfold the kpost forall+ into the gathered frac
 * quantifier and the acc2(scan2d_inclusive_result)-shaped cell
 * quantifier expected by [iraise]. *)
ghost
fn teardown_unfold_kpost_step
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (#rows : nat) (#cols : nat)
  (#lin  : layout2 rows cols)
  (#lout : layout2 rows cols)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx    : chest2 et rows cols)
  (#fIn   : perm)
  (#_ : squash (rows >= 1 /\ cols >= 1))
  ()
  norewrite
  requires
    forall+ (tid : natlt (rows * cols)).
       kpost m input output sx fIn tid
  ensures
    pure (SZ.fits (tlayout_ulen lout)) **
    (forall+ (_ : natlt (rows * cols)).
       input |-> Frac (fIn /. (rows * cols)) sx) **
    (forall+ (r : natlt rows) (c : natlt cols).
       tensor_pts_to_cell output (idx2 r c)
         (acc2 (scan2d_inclusive_result m sx) r c))
{
  (* 1. Split the (unfolded) kpost forall+ into two forall+s. *)
  forevery_unzip #(natlt (rows * cols))
    (fun (_ : natlt (rows * cols)) -> input |-> Frac (fIn /. (rows * cols)) sx)
    (fun (tid : natlt (rows * cols)) ->
       tensor_pts_to_cell output (tid_to_rc rows cols tid)
         (scan_inclusive_at m (ematrix_row sx (tid / cols)) (tid % cols)));
  (* 2. Reindex the cell quantifier from [tid] to [(r, c)]. *)
  forevery_factor' (rows * cols) rows cols
    (fun (r : natlt rows) (c : natlt cols) ->
       tensor_pts_to_cell output (idx2 r c)
         (scan_inclusive_at m (ematrix_row sx r) c));
  (* 3. Rewrite [scan_inclusive_at ...] to [acc2 (scan2d_inclusive_result ...) r c]. *)
  Classical.forall_intro_2
    (macc_scan2d_inclusive_result #et m #rows #cols sx);
  forevery_ext_2
    (fun (r : natlt rows) (c : natlt cols) ->
       tensor_pts_to_cell output (idx2 r c)
         (scan_inclusive_at m (ematrix_row sx r) c))
    (fun (r : natlt rows) (c : natlt cols) ->
       tensor_pts_to_cell output (idx2 r c)
         (acc2 (scan2d_inclusive_result m sx) r c));
  (* 4. Extract [pure (SZ.fits (layout_size lout))] from a single cell. *)
  forevery_extract_2 #(natlt rows) #(natlt cols) (0 <: natlt rows) (0 <: natlt cols)
    (fun (r : natlt rows) (c : natlt cols) ->
       tensor_pts_to_cell output (idx2 r c)
         (acc2 (scan2d_inclusive_result m sx) r c));
  pts_to_cell_fits #et #rows #cols #lout output (ait_zero rows cols ())
    #1.0R #(acc2 (scan2d_inclusive_result m sx) 0 0);
  Pulse.Lib.Trade.elim_trade _ _;
  ()
}

ghost
fn setup_scan
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v rows * SZ.v cols <= max_blocks * max_threads })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout2 (SZ.v rows) (SZ.v cols))
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et (SZ.v rows) (SZ.v cols))
  (#sout : chest2 et (SZ.v rows) (SZ.v cols))
  (#fIn  : perm)
  ()
  norewrite
  requires
    input |-> Frac fIn sx ** output |-> sout
  ensures
    forall+ (tid : natlt (SZ.v rows * SZ.v cols)).
      kpre m input output sx sout fIn tid
{
  tensor_share_n input (SZ.v rows * SZ.v cols);
  tensor_ilower2 output;
  setup_fold_kpre_step #et #_ m
    #(SZ.v rows) #(SZ.v cols)
    input output ();
  ()
}

ghost
fn teardown_scan
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v rows * SZ.v cols <= max_blocks * max_threads })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout2 (SZ.v rows) (SZ.v cols))
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et (SZ.v rows) (SZ.v cols))
  (#fIn  : perm)
  ()
  norewrite
  requires
    forall+ (tid : natlt (SZ.v rows * SZ.v cols)).
      kpost m input output sx fIn tid
  ensures
    input |-> Frac fIn sx **
    output |-> scan2d_inclusive_result m sx
{
  teardown_unfold_kpost_step #et #_ m
    #(SZ.v rows) #(SZ.v cols)
    input output #sx #fIn #() ();
  tensor_gather_n input (SZ.v rows * SZ.v cols);
  tensor_iraise2 output;
  ()
}

(* ──────────────────────────────────────────────────────────────────────
 * Section 4b: kdesc setup/teardown bridges
 *
 * The [kernel_desc_n] [setup]/[teardown] fields index the per-thread
 * [forall+] by [natlt (SZ.v (rows *^ cols))] (the size_t product used for
 * [nthr]), whereas the verified helpers [setup_scan] / [teardown_scan]
 * produce/consume it indexed by the multiplied-out nat
 * [natlt (SZ.v rows * SZ.v cols)].  These bounds are provably equal (the
 * [( *^ )] spec under no-overflow, which holds because
 * [SZ.v rows * SZ.v cols <= max_blocks * max_threads]).  [forevery_map']
 * rebinds the [forall+] across the type equality, so the kdesc fields can
 * plug the helpers in directly with no proof holes. *)
ghost
fn kdesc_setup
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v rows * SZ.v cols <= max_blocks * max_threads })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout2 (SZ.v rows) (SZ.v cols))
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et (SZ.v rows) (SZ.v cols))
  (#sout : chest2 et (SZ.v rows) (SZ.v cols))
  (#fIn  : perm)
  ()
  norewrite
  requires
    input |-> Frac fIn sx ** output |-> sout
  ensures
    (forall+ (tid : natlt (SZ.v (rows *^ cols))).
       kpre m input output sx sout fIn tid) ** emp
{
  setup_scan m rows cols input output #sx #sout #fIn ();
  assert pure (SZ.v (rows *^ cols) == SZ.v rows * SZ.v cols);
  forevery_map'
    (fun (tid : natlt (SZ.v rows * SZ.v cols)) ->
       kpre m input output sx sout fIn tid)
    (fun (tid : natlt (SZ.v (rows *^ cols))) ->
       kpre m input output sx sout fIn tid)
    fn tid tid' {
       rewrite (kpre m input output sx sout fIn tid)
            as (kpre m input output sx sout fIn tid');
    };
  ()
}

ghost
fn kdesc_teardown
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v rows * SZ.v cols <= max_blocks * max_threads })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout2 (SZ.v rows) (SZ.v cols))
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et (SZ.v rows) (SZ.v cols))
  (#fIn  : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt (SZ.v (rows *^ cols))).
       kpost m input output sx fIn tid) ** emp
  ensures
    input |-> Frac fIn sx **
    output |-> scan2d_inclusive_result m sx
{
  assert pure (SZ.v (rows *^ cols) == SZ.v rows * SZ.v cols);
  forevery_map'
    (fun (tid : natlt (SZ.v (rows *^ cols))) ->
       kpost m input output sx fIn tid)
    (fun (tid : natlt (SZ.v rows * SZ.v cols)) ->
       kpost m input output sx fIn tid)
    fn tid tid' {
       rewrite (kpost m input output sx fIn tid)
            as (kpost m input output sx fIn tid');
    };
  teardown_scan m rows cols input output #sx #fIn ();
  ()
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp)
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout2 (SZ.v rows) (SZ.v cols))
  {| _ : ctlayout lin, _ : ctlayout lout |}
  (input  : array2 et lin  { is_global input  })
  (output : array2 et lout { is_global output })
  (#sx   : chest2 et (SZ.v rows) (SZ.v cols))
  (#sout : chest2 et (SZ.v rows) (SZ.v cols))
  (#fIn  : perm)
  (#_ : squash (SZ.v rows * SZ.v cols <= max_blocks * max_threads))
  (#_ : squash (SZ.fits (SZ.v rows * SZ.v cols)))
  : kernel_desc
      (requires
        input  |-> Frac fIn sx **
        output |-> sout)
      (ensures
        input  |-> Frac fIn sx **
        output |-> scan2d_inclusive_result m sx)
  = {
    nthr = rows *^ cols;
    f = (fun (gid : szlt (rows *^ cols)) ->
           kf m input output #sx #sout #fIn gid);
    frame = emp;
    teardown = kdesc_teardown m rows cols input output;
    setup    = kdesc_setup m rows cols input output;
    kpre  = (fun (tid : natlt (SZ.v rows * SZ.v cols)) ->
              kpre m input output sx sout fIn tid);
    kpost = (fun (tid : natlt (SZ.v rows * SZ.v cols)) ->
              kpost m input output sx fIn tid);
    kpre_sendable  = solve;
    kpost_sendable = solve;
  } <: kernel_desc_n _ _

(* ──────────────────────────────────────────────────────────────────────
 * Section 5: launcher (matches [val scan1d_inclusive] in the .fsti)
 * ────────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn scan1d_inclusive
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp)
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lout |}
  (input  : array2 et lin  { is_global input  })
  (output : array2 et lout { is_global output })
  (#sx   : chest2 et (SZ.v rows) (SZ.v cols))
  (#sout : chest2 et (SZ.v rows) (SZ.v cols))
  (#fIn  : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (SZ.v rows * SZ.v cols <= max_blocks * max_threads) **
    pure (SZ.fits (SZ.v rows * SZ.v cols))
  ensures
    on gpu_loc (output |-> scan2d_inclusive_result m sx)
{
  launch_sync (kdesc m rows cols input output)
}
