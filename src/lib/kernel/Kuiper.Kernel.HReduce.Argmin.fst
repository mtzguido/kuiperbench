module Kuiper.Kernel.HReduce.Argmin

#lang-pulse

open Kuiper
open Kuiper.Math.Fmin
open Kuiper.Tensor
open Kuiper.Bijection { ( =~ ) }
open Kuiper.Float32
module SZ = Kuiper.SizeT
module Seq = FStar.Seq
module U32 = FStar.UInt32
module I64 = FStar.Int64
module Cast = FStar.Int.Cast

(* ── Pure spec ─────────────────────────────────────────────────────── *)

(* Per-row partial argmin over an [(rows, cols)] row-major chest2.
   Returns the exact (idx, val) state after scanning columns [0..k).
   The strict-[lt] update is the operation performed by the kernel. *)

[@@"opaque_to_smt"]
let rec row_argmin_partial
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
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
  (sx : chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_argmin_partial sx r 0 == (0, pos_inf))
          [SMTPat (row_argmin_partial sx r 0)]
  = assert_norm (row_argmin_partial sx r 0 == (0, pos_inf))

#push-options "--fuel 2 --ifuel 1"
let row_argmin_partial_succ
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
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

(* The selected index is in the scanned prefix.  We deliberately do
   not identify the accumulator with [fmin]: Kuiper's abstract [lt]
   API currently exposes no law connecting the comparison to [fmin],
   especially in the presence of NaNs. *)
let rec row_argmin_idx_inv
  (#rows #cols : nat)
  (sx : chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures fst (row_argmin_partial sx r k) <=
                       (if k = 0 then 0 else k - 1))
          (decreases k)
  = if k = 0 then ()
    else begin
      row_argmin_idx_inv sx r (k - 1);
      row_argmin_partial_succ sx r (k - 1);
      let (bi_pre, bv_pre) = row_argmin_partial sx r (k - 1) in
      let v = acc2 sx r (k - 1) in
      if lt v bv_pre then ()
      else ()
    end

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
  (sx : chest2 f32 rows cols)
  (r : natlt rows)
  : GTot I64.t
  = row_argmin_idx_inv sx r cols;
    let bi = fst (row_argmin_partial sx r cols) in
    assert (bi <= (if SZ.v cols = 0 then 0 else SZ.v cols - 1));
    assert (bi < pow2 63);
    I64.int_to_t bi

unfold
let kpre_batched_argmin
  (rows : szp)
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (sx   : chest2 f32 rows cols)
  (sout : chest1 i64 rows)
  (r : natlt rows)
  : slprop
  = x |-> Frac (1.0R /. SZ.v rows) sx **
    Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout r

unfold
let kpost_batched_argmin
  (rows : szp)
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (sx   : chest2 f32 rows cols)
  (sout : chest1 i64 rows)
  (r : natlt rows)
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
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)             {| ctlayout lout |}
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (#sx   : chest2 f32 rows cols)
  (#sout : chest1 i64 rows)
  (gid : szlt rows)
  ()
  norewrite
  preserves gpu
  requires
    kpre_batched_argmin rows cols x output sx sout gid
  ensures
    kpost_batched_argmin rows cols x output sx sout gid
{
  unfold kpre_batched_argmin rows cols x output sx sout gid;

  let mut ci_ref : sz = 0sz;
  let mut bi_ref : sz = 0sz;
  let mut bv_ref : f32 = pos_inf;

  while (!ci_ref <^ cols)
    invariant exists* (ci_v : SZ.t) (bi_v : SZ.t) (bv_v : f32).
      ci_ref |-> ci_v **
      bi_ref |-> bi_v **
      bv_ref |-> bv_v **
      x |-> Frac (1.0R /. SZ.v rows) sx **
      Cell output (((SZ.v gid <: natlt rows), ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout gid **
      pure (SZ.v ci_v <= SZ.v cols /\
            (let (bi, bv) = row_argmin_partial sx gid ci_v in
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
  row_argmin_idx_inv sx gid cols;
  let final_bi_u32 : U32.t = SZ.sizet_to_u32 final_bi;
  let final_bi_i64 : I64.t = Cast.uint32_to_int64 final_bi_u32;
  assert pure (I64.v final_bi_i64 == SZ.v final_bi);
  assert pure (argmin_i64 cols sx gid == final_bi_i64);
  tensor_write_cell output ((gid <: szlt rows), ()) final_bi_i64;

  fold kpost_batched_argmin rows cols x output sx sout gid;
}
#pop-options

(* ── Whole-output spec (per-row argmin_partial → seq) ─────────────── *)

let seq_reduce_rows_argmin
  (#rows : nat) (cols : szp { SZ.v cols < pow2 63 })
  (sx : chest2 f32 rows cols)
  : GTot (Seq.lseq i64 rows)
  = Seq.init_ghost rows (fun r -> argmin_i64 cols sx r)

(* ── Ghost setup ───────────────────────────────────────────────────── *)

ghost
fn setup_batched_argmin
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp { SZ.v cols < pow2 63 })
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (#sx   : chest2 f32 rows cols)
  (#sout : chest1 i64 rows)
  ()
  norewrite
  requires
    x |-> sx ** output |-> sout
  ensures
    (forall+ (r : natlt rows). kpre_batched_argmin rows cols x output sx sout r) **
    pure (SZ.fits (tlayout_ulen lout))
{
  tensor_pts_to_ref output;
  tensor_share_n x rows;
  tensor_explode output;
  forevery_iso (abs_bij #rows) _;

  forevery_zip #(natlt rows)
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (r : natlt rows) -> Cell output (abs_bij.gg r) |-> acc sout (abs_bij.gg r));

  forevery_ext #(natlt rows)
    (fun (r : natlt rows) ->
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
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 f32 lin)
  (output : array1 i64 lout)
  (#sx   : chest2 f32 rows cols)
  (#sout : chest1 i64 rows)
  ()
  norewrite
  requires
    (forall+ (r : natlt rows). kpost_batched_argmin rows cols x output sx sout r) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows_argmin cols sx)
{
  forevery_ext #(natlt rows)
    (kpost_batched_argmin rows cols x output sx sout)
    (fun (r : natlt rows) ->
       x |-> Frac (1.0R /. SZ.v rows) sx **
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmin_i64 cols sx r);

  forevery_unzip #(natlt rows)
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (r : natlt rows) ->
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmin_i64 cols sx r);

  tensor_gather_n x rows;

  let sout' : chest1 i64 rows = hide (seq_to_chest1 (seq_reduce_rows_argmin cols sx));
  forevery_ext #(natlt rows)
    (fun (r : natlt rows) ->
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> argmin_i64 cols sx r)
    (fun (r : natlt rows) -> Cell output (abs_bij.gg r) |-> acc (reveal sout') (abs_bij.gg r));

  forevery_iso_back (abs_bij #rows)
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
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)             {| ctlayout lout |}
  (x      : array2 f32 lin  { is_global x      })
  (output : array1 i64 lout { is_global output })
  (#sx   : chest2 f32 rows cols)
  (#sout : chest1 i64 rows)
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
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)             {| ctlayout lout |}
  (x      : array2 f32 lin  { is_global x      })
  (output : array1 i64 lout { is_global output })
  (#sx   : chest2 f32 rows cols)
  (#sout : chest1 i64 rows)
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout)
  ensures
    on gpu_loc (output |-> seq_to_chest1 (seq_reduce_rows_argmin cols sx))
{
  launch_sync (kdesc_batched_argmin rows cols x output #sx #sout);
}
