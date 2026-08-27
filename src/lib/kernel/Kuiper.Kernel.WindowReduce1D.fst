module Kuiper.Kernel.WindowReduce1D

(* Pulse implementation of the polymorphic 1-D windowed reduction primitive
 * declared in [Kuiper.Kernel.WindowReduce1D.fsti].
 *
 * For each output cell (r, j) the kernel computes
 *
 *   out[r, j] = m.rop-fold over the K-element OOB-filled dilated window
 *               at position j of input row r, with kernel/stride/pad/dilation
 *               (k, s, p, d).  Out-of-bounds positions are filled with [m.rid]
 *               so the seeded-fold result equals the fold over in-bounds
 *               elements alone (modulo the monoid laws).
 *
 * The primitive is polymorphic over [cmonoid t] so a single Pulse proof
 * covers both max-pool ([cmonoid_fmax_f32]) and avg-pool
 * ([cmonoid_fadd_f32]) instantiations.  The bridge from this fold-form
 * postcondition to the [Kuiper.Spec.Pool1D] [maxpool1d_post] /
 * [avgpool1d_post] predicates is delivered by per-challenge wrappers
 * (Kuiper.KB.{MaxPool*,AvgPool*}). *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Bijection
open Kuiper.Spec.Pool1D
open Kuiper.Monoid.Reduce
open Kuiper.Seq.Common
open Kuiper.EMatrix

module SZ = Kuiper.SizeT
module Seq = FStar.Seq

(* ──────────────────────────────────────────────────────────────────────
 * Section 1: prefix-fold lemmas
 *
 * The spec helpers [oob_window], [window_red], [ematrix_to_row],
 * [windowreduce_result] are defined in the .fsti so they appear in
 * the post.  This .fst adds the prefix-fold machinery used by the
 * per-thread loop invariant.
 * ────────────────────────────────────────────────────────────────────── *)

(* Partial fold over the first [di] taps (di ∈ [0, k]).  The per-thread
 * accumulator equals this value at iteration [di]. *)
let window_red_prefix
  (#t : Type0) (m : cmonoid t)
  (#l : nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat) (di : nat{di <= k})
  : GTot t
  = seq_fold_left m.rop m.rid (Seq.slice (oob_window m row k s p d j) 0 di)

let window_red_prefix_zero
  (#t : Type0) (m : cmonoid t)
  (#l : nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat)
  : Lemma (window_red_prefix m row k s p d j 0 == m.rid)
  = let w = oob_window m row k s p d j in
    let s0 = Seq.slice w 0 0 in
    Seq.lemma_eq_intro s0 Seq.empty;
    ()

(* Step lemma: extending the prefix by one tap appends the (di)-th OOB-filled
 * value and folds it in at the right end.  The key invariant-step. *)
#push-options "--z3rlimit 30"
let window_red_prefix_step
  (#t : Type0) (m : cmonoid t)
  (#l : nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat) (di : nat{di < k})
  : Lemma
      (let w = oob_window m row k s p d j in
       window_red_prefix m row k s p d j (di + 1)
       == m.rop (window_red_prefix m row k s p d j di) (Seq.index w di))
  = let w : Seq.lseq t k = oob_window m row k s p d j in
    let pre  : Seq.seq t = Seq.slice w 0 di in
    let pre' : Seq.seq t = Seq.slice w 0 (di + 1) in
    let one  : Seq.seq t = Seq.slice w di (di + 1) in
    Seq.lemma_eq_intro pre' (Seq.append pre one);
    Seq.lemma_eq_intro one (Seq.create 1 (Seq.index w di));
    red_fold_append m pre one;
    ()
#pop-options

(* The full fold equals the prefix at di == k. *)
let window_red_full
  (#t : Type0) (m : cmonoid t)
  (#l : nat) (row : Seq.lseq t l)
  (k s p d : nat) (j : nat)
  : Lemma (window_red m row k s p d j == window_red_prefix m row k s p d j k)
  = let w = oob_window m row k s p d j in
    Seq.lemma_eq_intro w (Seq.slice w 0 k);
    ()

(* ──────────────────────────────────────────────────────────────────────
 * Section 2: kpre / kpost predicates
 * ────────────────────────────────────────────────────────────────────── *)

unfold
let kpre
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (#rows : nat) (#l : nat) (#lo : nat)
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  (input  : array2 t lin)
  (output : array2 t lout)
  (sx   : chest2 t rows l)
  (sout : chest2 t rows lo)
  (fIn : perm)
  (gid : natlt (rows * lo))
  : slprop
  = input |-> Frac (fIn /. (rows * lo)) sx **
    tensor_pts_to_cell output (idx2 (gid / lo) (gid % lo))
      (acc2 sout (gid / lo) (gid % lo))

unfold
let kpost
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (k s p d : nat)
  (#rows : nat) (#l : nat) (#lo : nat)
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  (input  : array2 t lin)
  (output : array2 t lout)
  (sx   : chest2 t rows l)
  (fIn : perm)
  (gid : natlt (rows * lo))
  : slprop
  = input |-> Frac (fIn /. (rows * lo)) sx **
    tensor_pts_to_cell output (idx2 (gid / lo) (gid % lo))
      (window_red m (ematrix_to_row sx (gid / lo)) k s p d (gid % lo))

(* ──────────────────────────────────────────────────────────────────────
 * Section 3: per-thread function (the meat of the proof)
 *
 * Each thread owns one output cell (r, j) (decoded from gid = r*lo + j)
 * and a 1/(rows*lo) fraction of the whole input.  It runs a K-iteration
 * loop over the dilated taps, reading in-bounds taps from the input
 * (via [tensor_read_cell']) and folding them into [acc] with [m.rop],
 * then writes [acc] to the output cell.
 *
 * The loop invariant ties [acc] to [window_red_prefix m row k s p d j di]
 * via [window_red_prefix_step] applied at each iteration.
 * ────────────────────────────────────────────────────────────────────── *)


(* Per-thread predicate invariant: parametrized by [di] (loop counter). *)
let inv_at
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (k s p d : nat)
  (#rows : nat) (#l : nat) (#lo : nat)
  (sx : chest2 t rows l)
  (r : natlt rows) (j : natlt lo)
  (di : nat{di <= k}) (acc : t)
  : prop
  = acc == window_red_prefix m (ematrix_to_row sx r) k s p d j di

(* Convenience: SZ-fit precondition needed for the inner-loop arithmetic.
 * [j*s + (k-1)*d] must fit in [SZ.t]; we require the tighter
 * [SZ.fits (lo * s + k * d)] which subsumes it for any j < lo. *)
let sz_fits_window (k s p d lo : nat) : prop =
  SZ.fits (lo * s + k * d)

(* Helper: conditionally read one tap, returning [m.rid] for OOB taps. *)
inline_for_extraction noextract
fn read_tap
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (#rows : SZ.t) (#l : SZ.t)
  (#lin  : layout2 rows l)
  {| _ : ctlayout lin |}
  (input  : array2 et lin)
  (#sx    : chest2 et rows l)
  (#f     : perm)
  (r_sz : szlt rows)
  (raw  : SZ.t)
  (in_bounds : bool { in_bounds ==> SZ.v raw < SZ.v l })
  ()
  preserves
    input |-> Frac f sx
  returns v : et
  ensures
    pure (in_bounds ==> v == acc2 sx r_sz raw) **
    pure ((not in_bounds) ==> v == m.rid)
{
  if in_bounds {
    tensor_read input (cidx2 (r_sz <: szlt rows) (raw <: szlt l))
  } else {
    m.rid
  }
}

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (#rows : SZ.t) (#l : SZ.t) (#lo : SZ.t)
  (k s p d : SZ.t)
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  {| _ : ctlayout lin, _ : ctlayout lout |}
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et rows l)
  (#sout : chest2 et rows lo)
  (#fIn : perm)
  (#_ : squash (SZ.v k >= 1 /\ SZ.v s >= 1 /\ SZ.v d >= 1))
  (#_ : squash (SZ.v l >= 1))
  (#_ : squash (sz_fits_window k s p d lo))
  (gid : szlt (rows * lo))
  ()
  preserves gpu
  requires
    kpre m input output sx sout fIn gid
  ensures
    kpost m k s p d input output sx fIn gid
{
  let r_sz : szlt rows = gid /^ lo;
  let j_sz : szlt lo   = gid %^ lo;
  assert (pure (SZ.v r_sz == SZ.v gid / SZ.v lo /\ SZ.v j_sz == SZ.v gid % SZ.v lo));
  rewrite each (SZ.v gid / SZ.v lo) as r_sz;
  rewrite each (SZ.v gid % SZ.v lo) as j_sz;

  let row_view : erased (Seq.lseq et l) = ematrix_to_row sx r_sz;

  window_red_prefix_zero m row_view k s p d j_sz;

  let mut acc : et = m.rid;
  let mut di_ref : SZ.t = 0sz;
  let mut v_ref : et = m.rid;

  while (!di_ref <^ k)
    invariant exists* (di_v : SZ.t) (acc_v : et) (v_v : et).
      di_ref |-> di_v **
      acc |-> acc_v **
      v_ref |-> v_v **
      input |-> Frac (fIn /. (rows * lo)) sx **
      tensor_pts_to_cell output (idx2 (SZ.v r_sz <: natlt rows) (SZ.v j_sz <: natlt lo))
        (acc2 sout r_sz j_sz) **
      pure (SZ.v di_v <= SZ.v k /\
            acc_v == window_red_prefix m (reveal row_view) k s p d j_sz di_v)
    decreases (SZ.v k - SZ.v !di_ref)
  {
    let di_v : erased nat = SZ.v !di_ref;

    let js  : SZ.t = j_sz *^ s;
    let dd  : SZ.t = !di_ref *^ d;
    let pos : SZ.t = js +^ dd;

    let in_bounds : bool =
      pos >=^ p && (pos -^ p) <^ l;

    let mut dpos_ref : sz = 0sz;
    if in_bounds {
      dpos_ref := pos -^ p
    } else {
      ()
    };
    let dpos_safe : sz = !dpos_ref;
    assert pure (SZ.v dpos_safe < SZ.v l);
    let raw : et = tensor_read input (cidx2 (r_sz <: szlt rows) (dpos_safe <: szlt l));
    let v : et = if in_bounds { raw } else { m.rid };

    assert (pure (in_bounds <==>
            pool_in_bounds l s p d j_sz di_v));

    let w : erased (Seq.lseq et k) =
      oob_window m row_view k s p d j_sz;
    assert (pure (v == Seq.index w di_v));

    window_red_prefix_step m row_view k s p d j_sz di_v;
    acc := m.rop !acc v;
    di_ref := !di_ref +^ 1sz
  };

  window_red_full m row_view k s p d j_sz;
  let acc_final : et = !acc;
  assert pure (acc_final == window_red m (reveal row_view) k s p d j_sz);
  assert pure (reveal row_view == ematrix_to_row sx r_sz);
  tensor_write_cell output (cidx2 (r_sz <: szlt rows) (j_sz <: szlt lo)) acc_final;
  rewrite each (SZ.v r_sz) as (SZ.v gid / SZ.v lo);
  rewrite each (SZ.v j_sz) as (SZ.v gid % SZ.v lo);
  ()
}

(* ──────────────────────────────────────────────────────────────────────
 * Section 4: setup / teardown / kdesc
 *
 * The high-level structure of [setup] / [teardown] is verified Pulse
 * code: [share_n] / [gather_n] handle the input fraction, [ilower] /
 * [iraise] handle the output cells, and the acc2/window_red rewriting
 * for [iraise] is deferred to a per-cell SMT lemma.
 *
 * The leaf "fold/unfold the kpre/kpost forall+" step in each direction
 * is factored into a single named ghost-fn ([setup_fold_kpre_step] /
 * [teardown_unfold_kpost_step]).  Both have fully verified bodies built
 * from the [forevery_*] combinators (no proof holes).
 *
 *   - [setup_fold_kpre_step]
 *   - [teardown_unfold_kpost_step]
 *
 * The acc2/mk2 equality is captured separately in
 * [rewrite_macc_windowreduce_result] (a verified pure lemma).
 * ────────────────────────────────────────────────────────────────────── *)

(* Per-cell acc2 equality for [windowreduce_result].
 * Mathematically: by definition [windowreduce_result m sx k s p d lo]
 * is [mk2 (fun r c -> window_red m (ematrix_to_row sx r) k s p d c)],
 * so [acc2] of that at [(r, c)] reduces to the body. *)
let rewrite_macc_windowreduce_result
  (#t : Type0) {| scalar t |}
  (m : cmonoid t)
  (#rows #l : nat)
  (sx : chest2 t rows l)
  (k s p d lo : nat)
  (r : natlt rows) (c : natlt lo)
  : Lemma
      (acc2 (windowreduce_result m sx k s p d lo) r c ==
       window_red m (ematrix_to_row sx r) k s p d c)
  = ()

let tid_to_rc (rows lo : nat) (tid : natlt (rows * lo)) : abs (rows @| lo @| INil) =
  idx2 (tid / lo) (tid % lo)

let ait_zero (rows cols : nat) (_ : squash (rows >= 1 /\ cols >= 1))
  : abs (rows @| cols @| INil)
  = idx2 (0 <: natlt rows) (0 <: natlt cols)

(* Helper: a single [pts_to_cell] implies that the cell array's underlying
 * GPU array length [SZ.fits].  We unfold [pts_to_cell] to the underlying
 * [gpu_pts_to_slice] (via [pts_to_cell_eq]) and then use [gpu_pts_to_slice_ref]
 * to extract the [SZ.fits] fact. *)
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
       as pts_to_slice (core a) #f i (i + 1) (Seq.cons v Seq.empty);
  pts_to_slice_ref (core a) i (i + 1);
  rewrite pts_to_slice (core a) #f i (i + 1) (Seq.cons v Seq.empty)
       as tensor_pts_to_cell a #f ij v;
}

(* Setup leaf: fold the post-[share_n] frac quantifier and post-[ilower]
 * cell quantifier into the kpre forall+. *)
ghost
fn setup_fold_kpre_step
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (#rows : nat) (#l : nat) (#lo : nat)
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx    : chest2 et rows l)
  (#sout  : chest2 et rows lo)
  (#fIn   : perm)
  ()
  norewrite
  requires
    (forall+ (_ : natlt (rows * lo)).
       input |-> Frac (fIn /. (rows * lo)) sx) **
    (forall+ (r : natlt rows) (c : natlt lo).
       tensor_pts_to_cell output (idx2 r c) (acc2 sout r c))
  ensures
    forall+ (tid : natlt (rows * lo)).
      kpre m input output sx sout fIn tid
{
  forevery_unfactor' (rows * lo) rows lo
    (fun (r : natlt rows) (c : natlt lo) ->
       tensor_pts_to_cell output (idx2 r c) (acc2 sout r c));
  forevery_zip #(natlt (rows * lo))
    (fun (_ : natlt (rows * lo)) -> input |-> Frac (fIn /. (rows * lo)) sx)
    (fun (tid : natlt (rows * lo)) ->
       tensor_pts_to_cell output (tid_to_rc rows lo tid)
         (acc2 sout (tid / lo) (tid % lo)));
  forevery_ext _ (kpre m input output sx sout fIn);
  ()
}

(* Teardown leaf: unfold the kpost forall+ into the gathered frac
 * quantifier and the acc2(result)-shaped cell quantifier expected by
 * [iraise].  The extra [squash (rows * lo >= 1)] is needed solely to
 * extract a single cell and derive the [SZ.fits] fact required by
 * [iraise]; in the empty case [iraise] would not be called anyway. *)
ghost
fn teardown_unfold_kpost_step
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (k s p d : nat)
  (#rows : nat) (#l : nat) (#lo : nat)
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx    : chest2 et rows l)
  (#fIn   : perm)
  (#_ : squash (rows >= 1 /\ lo >= 1))
  ()
  norewrite
  requires
    forall+ (tid : natlt (rows * lo)).
       kpost m k s p d input output sx fIn tid
  ensures
    pure (SZ.fits (tlayout_ulen lout)) **
    (forall+ (_ : natlt (rows * lo)).
       input |-> Frac (fIn /. (rows * lo)) sx) **
    (forall+ (r : natlt rows) (c : natlt lo).
       tensor_pts_to_cell output (idx2 r c)
         (acc2 (windowreduce_result m sx k s p d lo) r c))
{
  (* 1. Split the (unfolded) kpost forall+ into two forall+s. *)
  forevery_unzip #(natlt (rows * lo))
    (fun (_ : natlt (rows * lo)) -> input |-> Frac (fIn /. (rows * lo)) sx)
    (fun (tid : natlt (rows * lo)) ->
       tensor_pts_to_cell output (tid_to_rc rows lo tid)
         (window_red m (ematrix_to_row sx (tid / lo)) k s p d (tid % lo)));
  (* 2. Reindex the cell quantifier from [tid] to [(r, c)]. *)
  forevery_factor' (rows * lo) rows lo
    (fun (r : natlt rows) (c : natlt lo) ->
       tensor_pts_to_cell output (idx2 r c)
         (window_red m (ematrix_to_row sx r) k s p d c));
  (* 3. Rewrite [window_red ...] to [acc2 (windowreduce_result ...) r c]. *)
  Classical.forall_intro_2
    (rewrite_macc_windowreduce_result #et #_ m #rows #l sx k s p d lo);
  forevery_ext_2
    (fun (r : natlt rows) (c : natlt lo) ->
       tensor_pts_to_cell output (idx2 r c)
         (window_red m (ematrix_to_row sx r) k s p d c))
    (fun (r : natlt rows) (c : natlt lo) ->
       tensor_pts_to_cell output (idx2 r c)
         (acc2 (windowreduce_result m sx k s p d lo) r c));
  (* 4. Extract [pure (SZ.fits (layout_size lout))] from a single cell. *)
  forevery_extract_2 #(natlt rows) #(natlt lo) (0 <: natlt rows) (0 <: natlt lo)
    (fun (r : natlt rows) (c : natlt lo) ->
       tensor_pts_to_cell output (idx2 r c)
         (acc2 (windowreduce_result m sx k s p d lo) r c));
  (* 5. Re-wrap the cell quantifier into the (forall+ r c) shape. *)
  pts_to_cell_fits #et #rows #lo #lout output (ait_zero rows lo ())
    #1.0R #(acc2 (windowreduce_result m sx k s p d lo) 0 0);
  Pulse.Lib.Trade.elim_trade _ _;
  ()
}

ghost
fn setup_windowreduce
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (l : SZ.t)
  (lo : szp { SZ.v rows * SZ.v lo <= max_blocks * max_threads })
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et rows l)
  (#sout : chest2 et rows lo)
  (#fIn  : perm)
  (#_ : squash (SZ.v rows >= 1 /\ SZ.v lo >= 1))
  ()
  norewrite
  requires
    input |-> Frac fIn sx ** output |-> sout
  ensures
    forall+ (tid : natlt (SZ.v rows * SZ.v lo)).
      kpre m input output sx sout fIn tid
{
  tensor_share_n input (SZ.v rows * SZ.v lo);
  tensor_ilower2 output;
  setup_fold_kpre_step #et #_ m
    #rows #l #lo
    input output ();
  ()
}

ghost
fn teardown_windowreduce
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (k s p d : SZ.t)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (l : SZ.t)
  (lo : szp { SZ.v rows * SZ.v lo <= max_blocks * max_threads })
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et rows l)
  (#fIn  : perm)
  (#_ : squash (SZ.v rows >= 1 /\ SZ.v lo >= 1))
  ()
  norewrite
  requires
    forall+ (tid : natlt (SZ.v rows * SZ.v lo)).
      kpost m k s p d input output sx fIn tid
  ensures
    input |-> Frac fIn sx **
    output |-> windowreduce_result m sx k s p d lo
{
  teardown_unfold_kpost_step #et #_ m
    k s p d
    #rows #l #lo
    input output #sx #fIn #() ();
  tensor_gather_n input (SZ.v rows * SZ.v lo);
  tensor_iraise2 output;
  ()
}

(* Adapt the setup and teardown helpers to the descriptor fields. *)
ghost
fn kdesc_setup
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (l : SZ.t)
  (lo : szp { SZ.v rows * SZ.v lo <= max_blocks * max_threads })
  (nthr : szp { SZ.v nthr == rows * lo })
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et rows l)
  (#sout : chest2 et rows lo)
  (#fIn  : perm)
  ()
  norewrite
  requires
    input |-> Frac fIn sx ** output |-> sout
  ensures
    (forall+ (tid : natlt nthr).
       kpre m input output sx sout fIn tid) ** emp
{
  setup_windowreduce m rows l lo input output #sx #sout #fIn #() ();
  forevery_rw_size (rows * lo) nthr;
  ()
}

ghost
fn kdesc_teardown
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (k s p d : SZ.t)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (l : SZ.t)
  (lo : szp { SZ.v rows * SZ.v lo <= max_blocks * max_threads })
  (nthr : szp { SZ.v nthr == rows * lo })
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  (input  : array2 et lin)
  (output : array2 et lout)
  (#sx   : chest2 et rows l)
  (#fIn  : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpost m k s p d input output sx fIn tid) ** emp
  ensures
    input |-> Frac fIn sx **
    output |-> windowreduce_result m sx k s p d lo
{
  forevery_rw_size nthr (rows * lo);
  teardown_windowreduce m k s p d rows l lo input output #sx #fIn #() ();
  ()
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (k s p d : SZ.t)
  (rows : szp)
  (l : SZ.t)
  (lo : szp {SZ.v lo == pool_out_len_1d l k s p d})
  (#lin  : layout2 rows l)
  (#lout : layout2 rows lo)
  {| _ : ctlayout lin, _ : ctlayout lout |}
  (input  : array2 et lin  { is_global input  })
  (output : array2 et lout { is_global output })
  (#sx   : chest2 et rows l)
  (#sout : chest2 et rows lo)
  (#fIn  : perm)
  (#_ : squash (SZ.v rows * SZ.v lo <= max_blocks * max_threads))
  (#_ : squash (SZ.v k >= 1 /\ SZ.v s >= 1 /\ SZ.v d >= 1))
  (#_ : squash (SZ.v l >= 1))
  (#_ : squash (sz_fits_window k s p d lo))
  : kernel_desc
      (requires
        input  |-> Frac fIn sx **
        output |-> sout)
      (ensures
        input  |-> Frac fIn sx **
        output |-> windowreduce_result m sx k s p d lo)
  = [@@inline_let] let nthr : (x : szp { SZ.v x == rows * lo }) = rows *^ lo in {
    nthr = nthr;
    f = (fun (gid : szlt nthr) ->
           kf m k s p d input output #sx
              #sout
              #fIn gid);
    frame = emp;
    teardown = kdesc_teardown m k s p d rows l lo nthr input output;
    setup    = kdesc_setup m rows l lo nthr input output;
    kpre  = (fun (tid : natlt (SZ.v rows * SZ.v lo)) ->
              kpre m input output sx sout fIn tid);
    kpost = (fun (tid : natlt (SZ.v rows * SZ.v lo)) ->
              kpost m k s p d input output sx
                    fIn tid);
    kpre_sendable  = solve;
    kpost_sendable = solve;
  } <: kernel_desc_n _ _

(* ──────────────────────────────────────────────────────────────────────
 * Section 5: launcher (matches the [val windowreduce] declaration in .fsti)
 * ────────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn windowreduce
  (#et : Type0) {| scalar et |}
  (m : cmonoid et)
  (k s : szp)
  (p : sz)
  (d : szp)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (l    : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
  (#lin  : layout2 rows l)     {| ctlayout lin  |}
  (#lout : layout2 rows l_out) {| ctlayout lout |}
  (input  : array2 et lin  { is_global input  })
  (output : array2 et lout { is_global output })
  (#sx   : chest2 et rows l)
  (#sout : chest2 et rows l_out)
  (#fIn  : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v rows * SZ.v l_out <= max_blocks * max_threads)
  ensures
    on gpu_loc (output |->
      windowreduce_result m sx k s p d l_out)
{
  if (l_out = 0sz) {
    (* Degenerate output (no columns): [sout] and the result matrix are
     * both [chest2 _ rows 0], hence extensionally equal (no cells).  The
     * [equal]/[ematrix_ext] SMTPats turn that into a propositional
     * equality, so the post follows from the precondition directly. *)
    assert pure (equal (reveal sout)
      (windowreduce_result m sx k s p d l_out));
    rewrite (on gpu_loc (output |-> sout))
         as (on gpu_loc (output |->
              windowreduce_result m sx k s p d l_out));
  } else {
    launch_sync (kdesc m k s p d rows l l_out input output)
  }
}
