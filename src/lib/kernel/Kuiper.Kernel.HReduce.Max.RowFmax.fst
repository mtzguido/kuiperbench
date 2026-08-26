module Kuiper.Kernel.HReduce.Max.RowFmax

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Math.Fmax
module EM = Kuiper.EMatrix
module Seq = FStar.Seq

(* ── Unfolding lemmas for the opaque [row_fmax_partial] ─────────────── *)

let row_fmax_partial_zero
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_fmax_partial sx r 0 == neg_inf)
          [SMTPat (row_fmax_partial sx r 0)]
  = assert_norm (row_fmax_partial sx r 0 == neg_inf)

#push-options "--fuel 2 --ifuel 1"
let row_fmax_partial_succ
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma (row_fmax_partial sx r (k + 1) ==
           fmax (row_fmax_partial sx r k) (acc2 sx r k))
          [SMTPat (row_fmax_partial sx r (k + 1))]
  = reveal_opaque (`%row_fmax_partial) (row_fmax_partial sx r (k + 1))
#pop-options

(* ── Bridge: [row_fmax_partial sx r cols == seq_fmax (ematrix_row sx r)] ─
   Proved by induction on [k = 0..cols] using [seq_fmax_append] on the
   one-element extension and the [seq_fmax_singleton] axiom. *)

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

let rec row_fmax_partial_eq_seq_fmax_aux
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  (k : nat{k <= cols})
  : Lemma (ensures row_fmax_partial sx r k == seq_fmax (row_prefix sx r k))
          (decreases k)
  = if k = 0 then begin
      row_fmax_partial_zero sx r;
      assert (Seq.equal (row_prefix sx r 0) Seq.empty);
      seq_fmax_empty ()
    end else begin
      row_fmax_partial_eq_seq_fmax_aux sx r (k - 1);
      row_prefix_succ sx r (k - 1);
      seq_fmax_append (row_prefix sx r (k - 1))
                      (Seq.create 1 (acc2 sx r (k - 1)));
      seq_fmax_singleton (acc2 sx r (k - 1));
      row_fmax_partial_succ sx r (k - 1)
    end

let row_fmax_eq_seq_fmax
  (#rows #cols : nat)
  (sx : EM.chest2 f32 rows cols)
  (r : natlt rows)
  : Lemma (row_fmax_partial sx r cols == seq_fmax (EM.ematrix_row sx r))
  = row_fmax_partial_eq_seq_fmax_aux sx r cols;
    row_prefix_full sx r;
    Seq.lemma_eq_intro (row_prefix sx r cols) (EM.ematrix_row sx r)
