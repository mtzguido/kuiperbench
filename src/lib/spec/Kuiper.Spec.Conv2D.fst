module Kuiper.Spec.Conv2D

open Kuiper
open Kuiper.EMatrix
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality
module Seq = FStar.Seq

(* Implementation of the Conv2D functional spec.  Mirrors the
   structure of [Kuiper.Spec.GEMM.fst]: a primitive recursive
   accumulator [__conv2d_single] over the linearised
   (ic, kh, kw) prefix, plus the obvious extensionality machinery
   for the 4D erased tensor type. *)

let tacc_pat (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (t : etensor4 et d0 d1 d2 d3)
  (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3)
  : Lemma (tacc t i j k l == t.f (i, j, k, l))
          [SMTPat (t.f (i, j, k, l))]
  = ()

let t4_equal #et #d0 #d1 #d2 #d3
  (t1 t2 : etensor4 et d0 d1 d2 d3) : prop
  = forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3).
      tacc t1 i j k l == tacc t2 i j k l

let lemma_t4_equal_intro #et #d0 #d1 #d2 #d3
  (t1 t2 : etensor4 et d0 d1 d2 d3)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3).
                      tacc t1 i j k l == tacc t2 i j k l)
          (ensures t4_equal t1 t2)
          [SMTPat (t4_equal t1 t2)]
  = ()

let etensor4_ext #et #d0 #d1 #d2 #d3
  (t1 t2 : etensor4 et d0 d1 d2 d3)
  : Lemma (requires t4_equal t1 t2)
          (ensures t1 == t2)
          [SMTPat (t4_equal t1 t2)]
  = let T4 f1 = t1 in
    let T4 f2 = t2 in
    let aux (idx : natlt d0 & natlt d1 & natlt d2 & natlt d3)
      : Lemma (f1 idx == f2 idx)
      = let (i, j, k, l) = idx in
        assert (tacc t1 i j k l == tacc t2 i j k l)
    in
    Classical.forall_intro aux;
    F.extensionality_g _ _ f1 f2

let rec __conv2d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (stride : pos)
  (pad : nat)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h_out)
  (ow : natlt w_out)
  (to : nat{to <= (if cin = 0 then 0 else cin * kh * kw)})
  : GTot et (decreases to)
  = if to = 0 then zero
    else (
      let i = to - 1 in
      assert (cin > 0);
      let ic : natlt cin = unrank_ic cin kh kw i in
      let kh_i : natlt kh = unrank_kh cin kh kw i in
      let kw_i : natlt kw = unrank_kw cin kh kw i in
      let h_idx : int = oh * stride + kh_i - pad in
      let w_idx : int = ow * stride + kw_i - pad in
      add
        (__conv2d_single kh kw stride pad x weight b oc oh ow (to - 1))
        (mul (read_padded x b ic h_idx w_idx)
             (tacc weight oc ic kh_i kw_i))
    )

let __conv2d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (stride : pos)
  (pad : nat)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h_out)
  (ow : natlt w_out)
  : Lemma
    (ensures __conv2d_single kh kw stride pad x weight b oc oh ow 0 == zero)
    [SMTPat (__conv2d_single kh kw stride pad x weight b oc oh ow 0)]
  = ()

let __conv2d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n : nat) (cin : pos) (#h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (stride : pos)
  (pad : nat)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h_out)
  (ow : natlt w_out)
  (to : pos{to <= cin * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank_ic cin kh kw i in
      let kh_i = unrank_kh cin kh kw i in
      let kw_i = unrank_kw cin kh kw i in
      let h_idx : int = oh * stride + kh_i - pad in
      let w_idx : int = ow * stride + kw_i - pad in
      __conv2d_single kh kw stride pad x weight b oc oh ow to ==
      add
        (__conv2d_single kh kw stride pad x weight b oc oh ow (to - 1))
        (mul (read_padded x b ic h_idx w_idx)
             (tacc weight oc ic kh_i kw_i))))
  = ()

let lemma_conv2d_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (stride : pos)
  (pad : nat)
  (h_out w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (oh : natlt h_out) (ow : natlt w_out)
  : Lemma (tacc (conv2d kh kw stride pad h_out w_out x weight bias) b oc oh ow
           == conv2d_single kh kw stride pad x weight bias b oc oh ow)
          [SMTPat (tacc (conv2d kh kw stride pad h_out w_out x weight bias)
                        b oc oh ow)]
  = ()
