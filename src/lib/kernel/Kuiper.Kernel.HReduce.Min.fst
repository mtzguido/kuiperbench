module Kuiper.Kernel.HReduce.Min

#lang-pulse

open Kuiper
open Kuiper.Math.Fmin
open Kuiper.Tensor
open Kuiper.Bijection { ( =~ ) }
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module Seq = FStar.Seq

(* ── Unfolding lemmas for the opaque [row_fmin_partial] ─────────────── *)

let row_fmin_partial_zero
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_fmin_partial sx r 0 == pos_inf)
          [SMTPat (row_fmin_partial sx r 0)]
  = assert_norm (row_fmin_partial sx r 0 == pos_inf)

#push-options "--fuel 2 --ifuel 1"
let row_fmin_partial_succ
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma (row_fmin_partial sx r (k + 1) ==
           fmin (row_fmin_partial sx r k) (acc2 sx r k))
          [SMTPat (row_fmin_partial sx r (k + 1))]
  = reveal_opaque (`%row_fmin_partial) (row_fmin_partial sx r (k + 1))
#pop-options

(* ── Bridge: [row_fmin_partial sx r cols == seq_fmin (ematrix_row sx r)] ─
   Proved by induction on [k = 0..cols] using [seq_fmin_append] on the
   one-element extension and the [seq_fmin_singleton] axiom. *)

let row_prefix
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : GTot (Seq.lseq f32 k)
  = Seq.init_ghost k (fun j -> acc2 sx r j)

let row_prefix_full
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (Seq.equal (row_prefix sx r cols) (EM.ematrix_row sx r))
  = ()

let row_prefix_succ
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma (Seq.equal
             (row_prefix sx r (k + 1))
             (Seq.append (row_prefix sx r k)
                         (Seq.create 1 (acc2 sx r k))))
  = ()

let rec row_fmin_partial_eq_seq_fmin_aux
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures row_fmin_partial sx r k == seq_fmin (row_prefix sx r k))
          (decreases k)
  = if k = 0 then begin
      row_fmin_partial_zero sx r;
      assert (Seq.equal (row_prefix sx r 0) Seq.empty);
      seq_fmin_empty ()
    end else begin
      row_fmin_partial_eq_seq_fmin_aux sx r (k - 1);
      row_prefix_succ sx r (k - 1);
      seq_fmin_append (row_prefix sx r (k - 1))
                      (Seq.create 1 (acc2 sx r (k - 1)));
      seq_fmin_singleton (acc2 sx r (k - 1));
      row_fmin_partial_succ sx r (k - 1)
    end

let row_fmin_eq_seq_fmin
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_fmin_partial sx r cols == seq_fmin (EM.ematrix_row sx r))
  = row_fmin_partial_eq_seq_fmin_aux sx r cols;
    row_prefix_full sx r;
    Seq.lemma_eq_intro (row_prefix sx r cols) (EM.ematrix_row sx r)

(* ── Per-thread predicates ─────────────────────────────────────────────── *)

(* Bijection between the abstract 1-D tensor index [(k, ())] and a plain
   [natlt len], used to (un)reindex a forevery over tensor cells. *)
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

unfold
let kpre_batched_min
  (rows : szp)
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout1 (SZ.v rows))
  (x      : array2 f32 lin)
  (output : array1 f32 lout)
  (sx   : erased (EM.chest2 f32 (SZ.v rows) (SZ.v cols)))
  (sout : chest1 f32 (SZ.v rows))
  (r : natlt (SZ.v rows))
  : slprop
  = x |-> Frac (1.0R /. SZ.v rows) sx **
    Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout r

unfold
let kpost_batched_min
  (rows : szp)
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout1 (SZ.v rows))
  (x      : array2 f32 lin)
  (output : array1 f32 lout)
  (sx   : erased (EM.chest2 f32 (SZ.v rows) (SZ.v cols)))
  (sout : chest1 f32 (SZ.v rows))
  (r : natlt (SZ.v rows))
  : slprop
  = x |-> Frac (1.0R /. SZ.v rows) sx **
    Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> row_fmin sx r

(* ── Per-thread kernel function ────────────────────────────────────────── *)

#push-options "--fuel 2 --ifuel 2 --z3rlimit 400"
inline_for_extraction noextract
fn kf_batched_min
  (rows : szp)
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : layout1 (SZ.v rows))             {| ctlayout lout |}
  (x      : array2 f32 lin)
  (output : array1 f32 lout)
  (#sx   : erased (EM.chest2 f32 (SZ.v rows) (SZ.v cols)))
  (#sout : chest1 f32 (SZ.v rows))
  (gid : szlt rows)
  ()
  norewrite
  requires
    gpu **
    kpre_batched_min rows cols x output sx sout (SZ.v gid)
  ensures
    gpu **
    kpost_batched_min rows cols x output sx sout (SZ.v gid)
{
  unfold kpre_batched_min rows cols x output sx sout (SZ.v gid);

  let mut ci_ref : sz = 0sz;
  let mut acc_ref : f32 = pos_inf;

  while (!ci_ref <^ cols)
    invariant exists* (ci_v : SZ.t) (acc_v : f32).
      ci_ref |-> ci_v **
      acc_ref |-> acc_v **
      x |-> Frac (1.0R /. SZ.v rows) sx **
      Cell output (((SZ.v gid <: natlt (SZ.v rows)), ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout (SZ.v gid) **
      pure (SZ.v ci_v <= SZ.v cols /\
            acc_v == row_fmin_partial sx (SZ.v gid) (SZ.v ci_v))
    decreases (SZ.v cols - SZ.v !ci_ref)
  {
    let ci_v_raw = !ci_ref;
    let ci_v : szlt cols = ci_v_raw;
    let v = tensor_read x (cidx2 gid ci_v);
    let acc_v = !acc_ref;
    assert pure (
      row_fmin_partial sx (SZ.v gid) (SZ.v ci_v + 1) ==
      fmin (row_fmin_partial sx (SZ.v gid) (SZ.v ci_v)) (acc2 sx (SZ.v gid) (SZ.v ci_v)));
    acc_ref := fmin acc_v v;
    ci_ref := !ci_ref +^ 1sz;
  };

  with acc_v. assert acc_ref |-> acc_v;
  let final_acc = !acc_ref;
  tensor_write_cell output ((gid <: szlt rows), ()) final_acc;

  fold kpost_batched_min rows cols x output sx sout (SZ.v gid);
}
#pop-options

(* ── Ghost setup ────────────────────────────────────────────────────────── *)

ghost
fn setup_batched_min
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout1 (SZ.v rows))
  (x      : array2 f32 lin)
  (output : array1 f32 lout)
  (#sx   : erased (EM.chest2 f32 (SZ.v rows) (SZ.v cols)))
  (#sout : chest1 f32 (SZ.v rows))
  ()
  norewrite
  requires
    x |-> sx ** output |-> sout
  ensures
    (forall+ (r : natlt (SZ.v rows)). kpre_batched_min rows cols x output sx sout r) **
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
    (kpre_batched_min rows cols x output sx sout);
  ()
}

(* ── Ghost teardown ─────────────────────────────────────────────────────── *)

ghost
fn teardown_batched_min
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols))
  (#lout : layout1 (SZ.v rows))
  (x      : array2 f32 lin)
  (output : array1 f32 lout)
  (#sx   : erased (EM.chest2 f32 (SZ.v rows) (SZ.v cols)))
  (#sout : chest1 f32 (SZ.v rows))
  ()
  norewrite
  requires
    (forall+ (r : natlt (SZ.v rows)). kpost_batched_min rows cols x output sx sout r) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows_fmin sx)
{
  forevery_ext #(natlt (SZ.v rows))
    (kpost_batched_min rows cols x output sx sout)
    (fun (r : natlt (SZ.v rows)) ->
       x |-> Frac (1.0R /. SZ.v rows) sx **
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> row_fmin sx r);

  forevery_unzip #(natlt (SZ.v rows))
    (fun (_ : natlt (SZ.v rows)) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (r : natlt (SZ.v rows)) -> Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> row_fmin sx r);

  tensor_gather_n x (SZ.v rows);

  let sout' : erased (chest1 f32 (SZ.v rows)) = hide (seq_to_chest1 (seq_reduce_rows_fmin sx));
  forevery_ext #(natlt (SZ.v rows))
    (fun (r : natlt (SZ.v rows)) -> Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> row_fmin sx r)
    (fun (r : natlt (SZ.v rows)) -> Cell output (abs_bij.gg r) |-> acc (reveal sout') (abs_bij.gg r));

  forevery_iso_back (abs_bij #(SZ.v rows))
    (fun (i : abs (SZ.v rows @| INil)) -> Cell output i |-> acc (reveal sout') i);

  tensor_implode output #1.0R #(reveal sout');
  ()
}

(* ── Kernel descriptor ──────────────────────────────────────────────────── *)

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let kdesc_batched_min
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : layout1 (SZ.v rows))             {| ctlayout lout |}
  (x      : array2 f32 lin  { is_global x      })
  (output : array1 f32 lout { is_global output })
  (#sx   : erased (EM.chest2 f32 (SZ.v rows) (SZ.v cols)))
  (#sout : chest1 f32 (SZ.v rows))
  : kernel_desc
      (x |-> sx ** output |-> sout)
      (x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows_fmin sx)) =
{
  nthr     = rows;
  frame    = pure (SZ.fits (tlayout_ulen lout));
  setup    = setup_batched_min rows cols x output;
  teardown = teardown_batched_min rows cols x output;
  kpre     = kpre_batched_min  rows cols x output sx sout;
  kpost    = kpost_batched_min rows cols x output sx sout;
  f        = kf_batched_min    rows cols x output;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _
#pop-options

(* ── Entry point ────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_min_f32
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin  |}
  (#lout : layout1 (SZ.v rows))             {| ctlayout lout |}
  (x      : array2 f32 lin  { is_global x      })
  (output : array1 f32 lout { is_global output })
  (#sx   : erased (EM.chest2 f32 (SZ.v rows) (SZ.v cols)))
  (#sout : chest1 f32 (SZ.v rows))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    on gpu_loc (output |-> sout)
  ensures
    on gpu_loc (x |-> sx) **
    on gpu_loc (output |-> seq_to_chest1 (seq_reduce_rows_fmin sx))
{
  launch_sync (kdesc_batched_min rows cols x output #sx #sout);
}
