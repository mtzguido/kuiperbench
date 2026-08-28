module Kuiper.Math.Fmin

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Seq.Common
module Seq = FStar.Seq

let rec seq_fold_left_append
  (#a #b : Type) (f : b -> a -> b) (acc : b)
  (l0 l1 : Seq.seq a)
  : Lemma (ensures seq_fold_left f acc (Seq.append l0 l1) ==
                   seq_fold_left f (seq_fold_left f acc l0) l1)
          (decreases Seq.length l0)
  = match view_seq l0 with
    | SNil -> assert (Seq.equal (Seq.append l0 l1) l1)
    | SCons hd tl ->
      let SCons hd' tl' = view_seq (Seq.append l0 l1) in
      assert (hd == hd');
      assert (Seq.equal tl' (Seq.append tl l1));
      seq_fold_left_append f (f acc hd) tl l1

let seq_fold_left_one
  (#a #b : Type) (f : b -> a -> b) (acc : b) (x : a)
  : Lemma (seq_fold_left f acc (Seq.create 1 x) == f acc x)
  = let s : Seq.seq a = Seq.create 1 x in
    assert (Seq.length s == 1);
    assert (Seq.head s == x);
    Seq.lemma_eq_intro (Seq.tail s) Seq.empty

let seq_fmin_snoc (s : Seq.seq f32) (x : f32)
  : Lemma (seq_fmin (Seq.snoc s x) == fmin (seq_fmin s) x)
  = seq_fold_left_append fmin pos_inf s (Seq.create 1 x);
    seq_fold_left_one fmin (seq_fmin s) x
