module Kuiper.Math.Argmax

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Math.Fmax
open Kuiper.Math.Fmin
module Seq = FStar.Seq

let gt_iff_fmax_strict (v w : f32)
  : Lemma (gt v w == true <==> (fmax v w == v /\ ~(v == w)))
  = admit()

let not_gt_fmax_keeps (v w : f32)
  : Lemma (gt v w == false ==> fmax v w == w)
  = admit()

let gt_neg_inf_or_eq (v : f32)
  : Lemma (gt v neg_inf == true \/ v == neg_inf)
  = admit()

let seq_fmax_geq (s : Seq.seq f32) (i : nat)
  : Lemma (requires i < Seq.length s)
          (ensures  gt (Seq.index s i) (seq_fmax s) == false)
  = admit()

let lt_iff_fmin_strict (v w : f32)
  : Lemma (lt v w == true <==> (fmin v w == v /\ ~(v == w)))
  = admit()

let not_lt_fmin_keeps (v w : f32)
  : Lemma (lt v w == false ==> fmin v w == w)
  = admit()

let lt_pos_inf_or_eq (v : f32)
  : Lemma (lt v pos_inf == true \/ v == pos_inf)
  = admit()

let seq_fmin_leq (s : Seq.seq f32) (i : nat)
  : Lemma (requires i < Seq.length s)
          (ensures  lt (Seq.index s i) (seq_fmin s) == false)
  = admit()
