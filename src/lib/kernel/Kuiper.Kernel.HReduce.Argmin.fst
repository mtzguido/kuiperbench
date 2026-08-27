module Kuiper.Kernel.HReduce.Argmin

#lang-pulse

open Kuiper
open Kuiper.Math.Fmin
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
module HRMin = Kuiper.Kernel.HReduce.Min

(* ── Pure spec ─────────────────────────────────────────────────────── *)

(* Per-row partial argmin over an [(rows, cols)] row-major chest2.
   Returns the (idx, val) pair after scanning columns [0..k).  The
   strict-[lt] update mirrors PyTorch's "first occurrence"
   convention; we verify the weaker "is_a_max" property. *)

[@@"opaque_to_smt"]
let rec row_argmin_partial
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot (nat & f32)
  (decreases k)
  = if k = 0 then (0, pos_inf)
    else
      let (bi, bv) = row_argmin_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      if lt v bv then (k - 1, v) else (bi, bv)

let row_argmin_partial_zero
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_argmin_partial sx r 0 == (0, pos_inf))
          [SMTPat (row_argmin_partial sx r 0)]
  = assert_norm (row_argmin_partial sx r 0 == (0, pos_inf))

#push-options "--fuel 2 --ifuel 1"
let row_argmin_partial_succ
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma
      (let (bi, bv) = row_argmin_partial sx r k in
       let v = acc2 sx r k in
       row_argmin_partial sx r (k + 1) ==
         (if lt v bv then (k, v) else (bi, bv)))
      [SMTPat (row_argmin_partial sx r (k + 1))]
  = reveal_opaque (`%row_argmin_partial) (row_argmin_partial sx r (k + 1))
#pop-options

(* The val component of [row_argmin_partial] equals [row_fmin_partial]
   from the verified Max primitive. *)
let rec row_argmin_val_eq_fmin
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures snd (row_argmin_partial sx r k) ==
                   HRMin.row_fmin_partial sx r k)
          (decreases k)
  = if k = 0 then ()
    else begin
      row_argmin_val_eq_fmin sx r (k - 1);
      HRMin.row_fmin_partial_succ sx r (k - 1);
      let (bi, bv) = row_argmin_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      let _ = fmin_comm in
      assert (bv == HRMin.row_fmin_partial sx r (k - 1));
      assert (HRMin.row_fmin_partial sx r k == fmin bv v);
      assert (fmin bv v == fmin v bv);
      if lt v bv then begin
        lt_iff_fmin_strict v bv;
        assert (fmin v bv == v);
        assert (snd (row_argmin_partial sx r k) == v)
      end else begin
        not_lt_fmin_keeps v bv;
        assert (fmin v bv == bv);
        assert (snd (row_argmin_partial sx r k) == bv)
      end
    end

(* The selected idx is in range and points to its value. *)
let rec row_argmin_idx_inv
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures (let (bi, bv) = row_argmin_partial sx r k in
                    bi <= (if k = 0 then 0 else k - 1) /\
                    (k > 0 ==> acc2 sx r bi == bv)))
          (decreases k)
  = if k = 0 then ()
    else if k = 1 then begin
      row_argmin_partial_succ sx r 0;
      assert (row_argmin_partial sx r 0 == (0, pos_inf));
      let v0 = acc2 sx r 0 in
      lt_pos_inf_or_eq v0;
      if lt v0 pos_inf then begin
        assert (row_argmin_partial sx r 1 == (0, v0))
      end else begin
        assert (row_argmin_partial sx r 1 == (0, pos_inf));
        assert (v0 == pos_inf)
      end
    end
    else begin
      row_argmin_idx_inv sx r (k - 1);
      row_argmin_partial_succ sx r (k - 1);
      let (bi_pre, bv_pre) = row_argmin_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      if lt v bv_pre then ()
      else ()
    end

(* Local prefix machinery (mirrors the un-exported helpers in the Min
   primitive) used to lift the [seq_fmin_leq] axiom to running-min
   prefixes. *)
let arg_row_prefix
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot (Seq.lseq f32 k)
  = Seq.init_ghost k (fun j -> acc2 sx r j)

let rec arg_row_fmin_eq
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures HRMin.row_fmin_partial sx r k == seq_fmin (arg_row_prefix sx r k))
          (decreases k)
  = if k = 0 then begin
      HRMin.row_fmin_partial_zero sx r;
      assert (Seq.equal (arg_row_prefix sx r 0) Seq.empty);
      seq_fmin_empty ()
    end else begin
      arg_row_fmin_eq sx r (k - 1);
      assert (Seq.equal (arg_row_prefix sx r k)
                (Seq.append (arg_row_prefix sx r (k - 1))
                            (Seq.create 1 (acc2 sx r (k - 1)))));
      seq_fmin_append (arg_row_prefix sx r (k - 1))
                      (Seq.create 1 (acc2 sx r (k - 1)));
      seq_fmin_singleton (acc2 sx r (k - 1));
      HRMin.row_fmin_partial_succ sx r (k - 1)
    end

(* First-occurrence invariant: every column strictly *before* the
   selected index has a value different from the running minimum.
   This is what the strict-[lt] update buys us — it pins the PyTorch
   "first occurrence" tie-break.  Proved by induction on [k]: in the
   update branch (new value [v] strictly smaller than the running min
   [bv]), every earlier column is [>= bv > v] so cannot equal [v]; in
   the keep branch the property is inherited unchanged. *)
let rec row_argmin_first_inv
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures (let (bi, bv) = row_argmin_partial sx r k in
                    bi <= (if k = 0 then 0 else k - 1) /\
                    (forall (j : nat). j < bi ==> ~(acc2 sx r j == bv))))
          (decreases k)
  = if k = 0 then ()
    else begin
      row_argmin_first_inv sx r (k - 1);
      row_argmin_idx_inv sx r (k - 1);
      row_argmin_partial_succ sx r (k - 1);
      let (bi, bv) = row_argmin_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      if lt v bv then begin
        (* new partial = (k-1, v); show no earlier column equals v *)
        row_argmin_val_eq_fmin sx r (k - 1);  (* bv == row_fmin_partial (k-1) *)
        arg_row_fmin_eq sx r (k - 1);         (* == seq_fmin (prefix (k-1)) *)
        let aux (j : nat{j < k - 1}) : Lemma (~(acc2 sx r j == v)) =
          (* each earlier value is not strictly below the running min bv;
             but v is, so they differ. *)
          seq_fmin_leq (arg_row_prefix sx r (k - 1)) j
        in
        Classical.forall_intro aux
      end else ()
    end

(* Bridge to is_a_max: at k=cols, the selected idx points at the max
   value of the full row.  Full functional bridge to seq_fmin via the
   Max primitive's row_fmin_eq_seq_fmin. *)
let row_argmin_at_full
  (#rows : nat) (#cols : nat{cols > 0})
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (let (bi, bv) = row_argmin_partial sx r cols in
           bi < cols /\
           bv == seq_fmin (EM.ematrix_row sx r) /\
           acc2 sx r bi == seq_fmin (EM.ematrix_row sx r))
  = row_argmin_idx_inv sx r cols;
    row_argmin_val_eq_fmin sx r cols;
    HRMin.row_fmin_eq_seq_fmin sx r

(* Strong full bridge: the selected idx is the *first* row-min, i.e. it
   points at [seq_fmin] and no strictly-earlier column attains it.  This
   is exactly the PyTorch first-occurrence argmin tie-break. *)
let row_argmin_first_at_full
  (#rows : nat) (#cols : nat{cols > 0})
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (let (bi, bv) = row_argmin_partial sx r cols in
           bi < cols /\
           acc2 sx r bi == seq_fmin (EM.ematrix_row sx r) /\
           (forall (j : nat). j < bi ==>
              ~(acc2 sx r j == seq_fmin (EM.ematrix_row sx r))))
  = row_argmin_at_full sx r;
    row_argmin_first_inv sx r cols

(* ── Per-thread predicates ─────────────────────────────────────────── *)

(* Bijection between the abstract 1-D tensor index [(k, ())] and a plain
   [natlt len], used to (un)reindex a forevery over tensor cells. *)
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

(* Coerce the partial-argmin index to an i64.  The index is always
   [<= max 0 (cols - 1)], which fits in i64 whenever cols comes from a
   szp (so [SZ.v cols < 2^32 < 2^63]). *)
noextract
let argmin_i64
  (#rows : nat) (cols : szp { SZ.v cols < pow2 63 })
  (sx : EM.chest2 f32 rows (SZ.v cols))
  (r : natlt rows)
  : GTot I64.t
  = row_argmin_idx_inv sx r (SZ.v cols);
    let bi = fst (row_argmin_partial sx r (SZ.v cols)) in
    assert (bi <= (if SZ.v cols = 0 then 0 else SZ.v cols - 1));
    assert (bi < pow2 63);
    I64.int_to_t bi

unfold
let kpre_batched_argmin
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
let kpost_batched_argmin
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
      argmin_i64 cols sx r

(* ── Per-thread kernel ─────────────────────────────────────────────── *)

#push-options "--fuel 2 --ifuel 2 --z3rlimit 400"
inline_for_extraction noextract
fn kf_batched_argmin
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
    kpre_batched_argmin rows cols x output sx sout (SZ.v gid)
  ensures
    gpu **
    kpost_batched_argmin rows cols x output sx sout (SZ.v gid)
{
  unfold kpre_batched_argmin rows cols x output sx sout (SZ.v gid);

  let mut ci_ref : sz = 0sz;
  let mut bi_ref : sz = 0sz;
  let mut bv_ref : f32 = pos_inf;

  while (!ci_ref <^ cols)
    invariant exists* (ci_v : SZ.t) (bi_v : SZ.t) (bv_v : f32).
      ci_ref |-> ci_v **
      bi_ref |-> bi_v **
      bv_ref |-> bv_v **
      x |-> Frac (1.0R /. SZ.v rows) sx **
      Cell output (((SZ.v gid <: natlt (SZ.v rows)), ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout (SZ.v gid) **
      pure (SZ.v ci_v <= SZ.v cols /\
            (let (bi, bv) = row_argmin_partial sx (SZ.v gid) (SZ.v ci_v) in
             SZ.v bi_v == bi /\ bv_v == bv))
    decreases (SZ.v cols - SZ.v !ci_ref)
  {
    let ci_v_raw = !ci_ref;
    let ci_v : szlt cols = ci_v_raw;
    let v = tensor_read x (cidx2 gid ci_v);
    let bv_v = !bv_ref;
    if lt v bv_v {
      bv_ref := v;
      bi_ref := ci_v;
    } else {
      ()
    };
    ci_ref := !ci_ref +^ 1sz;
  };

  with bi_v. assert bi_ref |-> bi_v;
  let final_bi = !bi_ref;
  (* SZ.v final_bi < SZ.v cols by row_argmin_idx_inv ; cast through u32 → i64 *)
  row_argmin_idx_inv sx (SZ.v gid) (SZ.v cols);
  let final_bi_u32 : U32.t = SZ.sizet_to_u32 final_bi;
  let final_bi_i64 : I64.t = Cast.uint32_to_int64 final_bi_u32;
  assert pure (I64.v final_bi_i64 == SZ.v final_bi);
  assert pure (argmin_i64 cols sx (SZ.v gid) == final_bi_i64);
  tensor_write_cell output ((gid <: szlt rows), ()) final_bi_i64;

  fold kpost_batched_argmin rows cols x output sx sout (SZ.v gid);
}
#pop-options

(* ── Whole-output spec (per-row argmin_partial → seq) ─────────────── *)

let seq_reduce_rows_argmin
  (#rows : nat) (cols : szp { SZ.v cols < pow2 63 })
  (sx : EM.chest2 f32 rows (SZ.v cols))
  : GTot (Seq.lseq i64 rows)
  = Seq.init_ghost rows (fun r -> argmin_i64 cols sx r)

(* ── Ghost setup ───────────────────────────────────────────────────── *)

ghost
fn setup_batched_argmin
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
    (forall+ (r : natlt (SZ.v rows)). kpre_batched_argmin rows cols x output sx sout r) **
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
    (kpre_batched_argmin rows cols x output sx sout);
  ()
}

(* ── Ghost teardown ────────────────────────────────────────────────── *)

ghost
fn teardown_batched_argmin
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
    (forall+ (r : natlt (SZ.v rows)). kpost_batched_argmin rows cols x output sx sout r) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows_argmin cols sx)
{
  forevery_ext #(natlt (SZ.v rows))
    (kpost_batched_argmin rows cols x output sx sout)
    (fun (r : natlt (SZ.v rows)) ->
       x |-> Frac (1.0R /. SZ.v rows) sx **
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmin_i64 cols sx r);

  forevery_unzip #(natlt (SZ.v rows))
    (fun (_ : natlt (SZ.v rows)) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (r : natlt (SZ.v rows)) ->
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmin_i64 cols sx r);

  tensor_gather_n x (SZ.v rows);

  let sout' : chest1 i64 (SZ.v rows) = hide (seq_to_chest1 (seq_reduce_rows_argmin cols sx));
  forevery_ext #(natlt (SZ.v rows))
    (fun (r : natlt (SZ.v rows)) ->
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmin_i64 cols sx r)
    (fun (r : natlt (SZ.v rows)) -> Cell output (abs_bij.gg r) |-> acc (reveal sout') (abs_bij.gg r));

  forevery_iso_back (abs_bij #(SZ.v rows))
    (fun (i : abs (SZ.v rows @| INil)) -> Cell output i |-> acc (reveal sout') i);

  tensor_implode output #1.0R #(reveal sout');
  ()
}

(* ── Kernel descriptor ─────────────────────────────────────────────── *)

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let kdesc_batched_argmin
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
      (x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows_argmin cols sx)) =
{
  nthr     = rows;
  frame    = pure (SZ.fits (tlayout_ulen lout));
  setup    = setup_batched_argmin rows cols x output;
  teardown = teardown_batched_argmin rows cols x output;
  kpre     = kpre_batched_argmin  rows cols x output sx sout;
  kpost    = kpost_batched_argmin rows cols x output sx sout;
  f        = kf_batched_argmin    rows cols x output;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _
#pop-options

(* ── Entry point ──────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_argmin_f32
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
    on gpu_loc (output |-> seq_to_chest1 (seq_reduce_rows_argmin cols sx))
{
  launch_sync (kdesc_batched_argmin rows cols x output #sx #sout);
}
