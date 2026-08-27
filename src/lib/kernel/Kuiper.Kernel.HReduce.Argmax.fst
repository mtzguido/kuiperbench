module Kuiper.Kernel.HReduce.Argmax

#lang-pulse

open Kuiper
open Kuiper.Math.Fmax
open Kuiper.Math.Argmax
open Kuiper.Tensor
open Kuiper.Bijection { ( =~ ) }
open Kuiper.Float32
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module Seq = FStar.Seq
module U32 = FStar.UInt32
module I64 = FStar.Int64
module Cast = FStar.Int.Cast
module HRM = Kuiper.Kernel.HReduce.Max.RowFmax

(* ── Pure spec ─────────────────────────────────────────────────────── *)

(* Per-row partial argmax over an [(rows, cols)] row-major chest2.
   Returns the (idx, val) pair after scanning columns [0..k).  The
   strict-[gt] update mirrors PyTorch's "first occurrence"
   convention; we verify the weaker "is_a_max" property. *)

[@@"opaque_to_smt"]
let rec row_argmax_partial
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot (nat & f32)
  (decreases k)
  = if k = 0 then (0, neg_inf)
    else
      let (bi, bv) = row_argmax_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      if gt v bv then (k - 1, v) else (bi, bv)

let row_argmax_partial_zero
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_argmax_partial sx r 0 == (0, neg_inf))
          [SMTPat (row_argmax_partial sx r 0)]
  = assert_norm (row_argmax_partial sx r 0 == (0, neg_inf))

#push-options "--fuel 2 --ifuel 1"
let row_argmax_partial_succ
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma
      (let (bi, bv) = row_argmax_partial sx r k in
       let v = acc2 sx r k in
       row_argmax_partial sx r (k + 1) ==
         (if gt v bv then (k, v) else (bi, bv)))
      [SMTPat (row_argmax_partial sx r (k + 1))]
  = reveal_opaque (`%row_argmax_partial) (row_argmax_partial sx r (k + 1))
#pop-options

(* The val component of [row_argmax_partial] equals [row_fmax_partial]
   from the verified Max primitive. *)
let rec row_argmax_val_eq_fmax
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures snd (row_argmax_partial sx r k) ==
                   HRM.row_fmax_partial sx r k)
          (decreases k)
  = if k = 0 then ()
    else begin
      row_argmax_val_eq_fmax sx r (k - 1);
      HRM.row_fmax_partial_succ sx r (k - 1);
      let (bi, bv) = row_argmax_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      let _ = fmax_comm in
      assert (bv == HRM.row_fmax_partial sx r (k - 1));
      assert (HRM.row_fmax_partial sx r k == fmax bv v);
      assert (fmax bv v == fmax v bv);
      if gt v bv then begin
        gt_iff_fmax_strict v bv;
        assert (fmax v bv == v);
        assert (snd (row_argmax_partial sx r k) == v)
      end else begin
        not_gt_fmax_keeps v bv;
        assert (fmax v bv == bv);
        assert (snd (row_argmax_partial sx r k) == bv)
      end
    end

(* The selected idx is in range and points to its value. *)
let rec row_argmax_idx_inv
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures (let (bi, bv) = row_argmax_partial sx r k in
                    bi <= (if k = 0 then 0 else k - 1) /\
                    (k > 0 ==> acc2 sx r bi == bv)))
          (decreases k)
  = if k = 0 then ()
    else if k = 1 then begin
      row_argmax_partial_succ sx r 0;
      assert (row_argmax_partial sx r 0 == (0, neg_inf));
      let v0 = acc2 sx r 0 in
      gt_neg_inf_or_eq v0;
      if gt v0 neg_inf then begin
        assert (row_argmax_partial sx r 1 == (0, v0))
      end else begin
        assert (row_argmax_partial sx r 1 == (0, neg_inf));
        assert (v0 == neg_inf)
      end
    end
    else begin
      row_argmax_idx_inv sx r (k - 1);
      row_argmax_partial_succ sx r (k - 1);
      let (bi_pre, bv_pre) = row_argmax_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      if gt v bv_pre then ()
      else ()
    end

(* Local prefix machinery used to lift the [seq_fmax_geq] axiom to
   running-max prefixes (mirrors the un-exported helpers in RowFmax). *)
let arg_row_prefix
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot (Seq.lseq f32 k)
  = Seq.init_ghost k (fun j -> acc2 sx r j)

let rec arg_row_fmax_eq
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures HRM.row_fmax_partial sx r k == seq_fmax (arg_row_prefix sx r k))
          (decreases k)
  = if k = 0 then begin
      HRM.row_fmax_partial_zero sx r;
      assert (Seq.equal (arg_row_prefix sx r 0) Seq.empty);
      seq_fmax_empty ()
    end else begin
      arg_row_fmax_eq sx r (k - 1);
      assert (Seq.equal (arg_row_prefix sx r k)
                (Seq.append (arg_row_prefix sx r (k - 1))
                            (Seq.create 1 (acc2 sx r (k - 1)))));
      seq_fmax_append (arg_row_prefix sx r (k - 1))
                      (Seq.create 1 (acc2 sx r (k - 1)));
      seq_fmax_singleton (acc2 sx r (k - 1));
      HRM.row_fmax_partial_succ sx r (k - 1)
    end

(* First-occurrence invariant: every column strictly *before* the
   selected index has a value different from the running maximum.
   This is what the strict-[gt] update buys us — it pins the PyTorch
   "first occurrence" tie-break.  Proved by induction on [k]: in the
   update branch (new value [v] strictly greater than the running max
   [bv]), every earlier column is [<= bv < v] so cannot equal [v]; in
   the keep branch the property is inherited unchanged. *)
let rec row_argmax_first_inv
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures (let (bi, bv) = row_argmax_partial sx r k in
                    bi <= (if k = 0 then 0 else k - 1) /\
                    (forall (j : nat). j < bi ==> ~(acc2 sx r j == bv))))
          (decreases k)
  = if k = 0 then ()
    else begin
      row_argmax_first_inv sx r (k - 1);
      row_argmax_idx_inv sx r (k - 1);
      row_argmax_partial_succ sx r (k - 1);
      let (bi, bv) = row_argmax_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      if gt v bv then begin
        (* new partial = (k-1, v); show no earlier column equals v *)
        row_argmax_val_eq_fmax sx r (k - 1);  (* bv == row_fmax_partial (k-1) *)
        arg_row_fmax_eq sx r (k - 1);         (* == seq_fmax (prefix (k-1)) *)
        let aux (j : nat{j < k - 1}) : Lemma (~(acc2 sx r j == v)) =
          (* each earlier value is not strictly above the running max bv;
             but v is, so they differ. *)
          seq_fmax_geq (arg_row_prefix sx r (k - 1)) j
        in
        Classical.forall_intro aux
      end else ()
    end

(* Bridge to is_a_max: at k=cols, the selected idx points at the max
   value of the full row.  Full functional bridge to seq_fmax via the
   Max primitive's row_fmax_eq_seq_fmax. *)
let row_argmax_at_full
  (#rows : nat) (#cols : nat{cols > 0})
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (let (bi, bv) = row_argmax_partial sx r cols in
           bi < cols /\
           bv == seq_fmax (EM.ematrix_row sx r) /\
           acc2 sx r bi == seq_fmax (EM.ematrix_row sx r))
  = row_argmax_idx_inv sx r cols;
    row_argmax_val_eq_fmax sx r cols;
    HRM.row_fmax_eq_seq_fmax sx r

(* Strong full bridge: the selected idx is the *first* row-max, i.e. it
   points at [seq_fmax] and no strictly-earlier column attains it.  This
   is exactly the PyTorch first-occurrence argmax tie-break. *)
let row_argmax_first_at_full
  (#rows : nat) (#cols : nat{cols > 0})
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (let (bi, bv) = row_argmax_partial sx r cols in
           bi < cols /\
           acc2 sx r bi == seq_fmax (EM.ematrix_row sx r) /\
           (forall (j : nat). j < bi ==>
              ~(acc2 sx r j == seq_fmax (EM.ematrix_row sx r))))
  = row_argmax_at_full sx r;
    row_argmax_first_inv sx r cols

(* ── Per-thread predicates ─────────────────────────────────────────── *)

(* Coerce the partial-argmax index to an i64.  The index is always
   [<= max 0 (cols - 1)], which fits in i64 whenever cols comes from a
   szp (so [SZ.v cols < 2^32 < 2^63]). *)
noextract
let argmax_i64
  (#rows : nat) (cols : szp { SZ.v cols < pow2 63 })
  (sx : EM.chest2 f32 rows (SZ.v cols))
  (r : natlt rows)
  : GTot I64.t
  = row_argmax_idx_inv sx r (SZ.v cols);
    let bi = fst (row_argmax_partial sx r (SZ.v cols)) in
    assert (bi <= (if SZ.v cols = 0 then 0 else SZ.v cols - 1));
    assert (bi < pow2 63);
    I64.int_to_t bi

(* Bijection between the abstract 1-D tensor index [(k, ())] and a plain
   [natlt len], used to (un)reindex a forevery over tensor cells. *)
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

unfold
let kpre_batched_argmax
  (rows : szp)
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout1 (SZ.v rows))
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (sx   : EM.chest2 f32 (SZ.v rows) (SZ.v cols))
  (sout : chest1 i64 (SZ.v rows))
  (r : natlt (SZ.v rows))
  : slprop
  = x |-> Frac (1.0R /. SZ.v rows) sx **
    Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout r

unfold
let kpost_batched_argmax
  (rows : szp)
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout1 (SZ.v rows))
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (sx   : EM.chest2 f32 (SZ.v rows) (SZ.v cols))
  (sout : chest1 i64 (SZ.v rows))
  (r : natlt (SZ.v rows))
  : slprop
  = x |-> Frac (1.0R /. SZ.v rows) sx **
    Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |->
      argmax_i64 cols sx r

(* ── Per-thread kernel ─────────────────────────────────────────────── *)

#push-options "--fuel 2 --ifuel 2 --z3rlimit 400"
inline_for_extraction noextract
fn kf_batched_argmax
  (rows : szp)
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : layout1 (SZ.v rows))             {| ctlayout lout |}
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (#sx   : EM.chest2 f32 (SZ.v rows) (SZ.v cols))
  (#sout : chest1 i64 (SZ.v rows))
  (gid : szlt rows)
  ()
  norewrite
  requires
    gpu **
    kpre_batched_argmax rows cols x output sx sout (SZ.v gid)
  ensures
    gpu **
    kpost_batched_argmax rows cols x output sx sout (SZ.v gid)
{
  unfold kpre_batched_argmax rows cols x output sx sout (SZ.v gid);

  let mut ci_ref : sz = 0sz;
  let mut bi_ref : sz = 0sz;
  let mut bv_ref : f32 = neg_inf;

  while (!ci_ref <^ cols)
    invariant exists* (ci_v : SZ.t) (bi_v : SZ.t) (bv_v : f32).
      ci_ref |-> ci_v **
      bi_ref |-> bi_v **
      bv_ref |-> bv_v **
      x |-> Frac (1.0R /. SZ.v rows) sx **
      Cell output (((SZ.v gid <: natlt (SZ.v rows)), ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout (SZ.v gid) **
      pure (SZ.v ci_v <= SZ.v cols /\
            (let (bi, bv) = row_argmax_partial sx (SZ.v gid) (SZ.v ci_v) in
             SZ.v bi_v == bi /\ bv_v == bv))
    decreases (SZ.v cols - SZ.v !ci_ref)
  {
    let ci_v_raw = !ci_ref;
    let ci_v : szlt cols = ci_v_raw;
    let v = tensor_read x (cidx2 gid ci_v);
    let bv_v = !bv_ref;
    if gt v bv_v {
      bv_ref := v;
      bi_ref := ci_v;
    } else {
      ()
    };
    ci_ref := !ci_ref +^ 1sz;
  };

  with bi_v. assert bi_ref |-> bi_v;
  let final_bi = !bi_ref;
  (* SZ.v final_bi < SZ.v cols by row_argmax_idx_inv ; cast through u32 → i64 *)
  row_argmax_idx_inv sx (SZ.v gid) (SZ.v cols);
  let final_bi_u32 : U32.t = SZ.sizet_to_u32 final_bi;
  let final_bi_i64 : I64.t = Cast.uint32_to_int64 final_bi_u32;
  assert pure (I64.v final_bi_i64 == SZ.v final_bi);
  assert pure (argmax_i64 cols sx (SZ.v gid) == final_bi_i64);
  tensor_write_cell output ((gid <: szlt rows), ()) final_bi_i64;

  fold kpost_batched_argmax rows cols x output sx sout (SZ.v gid);
}
#pop-options

(* ── Whole-output spec (per-row argmax_partial → seq) ─────────────── *)

let seq_reduce_rows_argmax
  (#rows : nat) (cols : szp { SZ.v cols < pow2 63 })
  (sx : EM.chest2 f32 rows (SZ.v cols))
  : GTot (Seq.lseq i64 rows)
  = Seq.init_ghost rows (fun r -> argmax_i64 cols sx r)

(* ── Ghost setup ───────────────────────────────────────────────────── *)

ghost
fn setup_batched_argmax
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout1 (SZ.v rows))
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (#sx   : EM.chest2 f32 (SZ.v rows) (SZ.v cols))
  (#sout : chest1 i64 (SZ.v rows))
  ()
  norewrite
  requires
    x |-> sx ** output |-> sout
  ensures
    (forall+ (r : natlt (SZ.v rows)). kpre_batched_argmax rows cols x output sx sout r) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_pts_to_ref output;
  tensor_share_n x (SZ.v rows);
  tensor_explode output;
  forevery_iso (abs_bij #(SZ.v rows)) _;

  forevery_zip #(natlt (SZ.v rows))
    (fun (_ : natlt (SZ.v rows)) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (r : natlt (SZ.v rows)) -> Cell output (abs_bij.gg r) |-> acc sout (abs_bij.gg r));

  forevery_ext #(natlt (SZ.v rows))
    (fun (r : natlt (SZ.v rows)) ->
       (x |-> Frac (1.0R /. SZ.v rows) (reveal sx)) **
       Cell output (abs_bij.gg r) |-> acc sout (abs_bij.gg r))
    (kpre_batched_argmax rows cols x output sx sout);
  ()
}

(* ── Ghost teardown ────────────────────────────────────────────────── *)

ghost
fn teardown_batched_argmax
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout1 (SZ.v rows))
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (#sx   : EM.chest2 f32 (SZ.v rows) (SZ.v cols))
  (#sout : chest1 i64 (SZ.v rows))
  ()
  norewrite
  requires
    (forall+ (r : natlt (SZ.v rows)). kpost_batched_argmax rows cols x output sx sout r) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows_argmax cols sx)
{
  forevery_ext #(natlt (SZ.v rows))
    (kpost_batched_argmax rows cols x output sx sout)
    (fun (r : natlt (SZ.v rows)) ->
       x |-> Frac (1.0R /. SZ.v rows) sx **
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmax_i64 cols sx r);

  forevery_unzip #(natlt (SZ.v rows))
    (fun (_ : natlt (SZ.v rows)) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (r : natlt (SZ.v rows)) ->
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmax_i64 cols sx r);

  tensor_gather_n x (SZ.v rows);

  let sout' : chest1 i64 (SZ.v rows) = hide (seq_to_chest1 (seq_reduce_rows_argmax cols sx));
  forevery_ext #(natlt (SZ.v rows))
    (fun (r : natlt (SZ.v rows)) ->
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmax_i64 cols sx r)
    (fun (r : natlt (SZ.v rows)) -> Cell output (abs_bij.gg r) |-> acc (reveal sout') (abs_bij.gg r));

  forevery_iso_back (abs_bij #(SZ.v rows))
    (fun (i : abs (SZ.v rows @| INil)) -> Cell output i |-> acc (reveal sout') i);

  tensor_implode output #1.0R #(reveal sout');
  ()
}

(* ── Kernel descriptor ─────────────────────────────────────────────── *)

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let kdesc_batched_argmax
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : layout1 (SZ.v rows))             {| ctlayout lout |}
  (x      : array2 f32 lin  { is_global x      })
  (output : array1 i64 lout { is_global output })
  (#sx   : EM.chest2 f32 (SZ.v rows) (SZ.v cols))
  (#sout : chest1 i64 (SZ.v rows))
  : kernel_desc
      (x |-> sx ** output |-> sout)
      (x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows_argmax cols sx)) =
{
  nthr     = rows;
  frame    = pure (SZ.fits (tlayout_ulen lout));
  setup    = setup_batched_argmax rows cols x output;
  teardown = teardown_batched_argmax rows cols x output;
  kpre     = kpre_batched_argmax  rows cols x output sx sout;
  kpost    = kpost_batched_argmax rows cols x output sx sout;
  f        = kf_batched_argmax    rows cols x output;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _
#pop-options

(* ── Entry point ──────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_argmax_f32
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : layout1 (SZ.v rows))             {| ctlayout lout |}
  (x      : array2 f32 lin  { is_global x      })
  (output : array1 i64 lout { is_global output })
  (#sx   : EM.chest2 f32 (SZ.v rows) (SZ.v cols))
  (#sout : chest1 i64 (SZ.v rows))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    on gpu_loc (output |-> sout)
  ensures
    on gpu_loc (x |-> sx) **
    on gpu_loc (output |-> seq_to_chest1 (seq_reduce_rows_argmax cols sx))
{
  launch_sync (kdesc_batched_argmax rows cols x output #sx #sout);
}
