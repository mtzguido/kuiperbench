module Kuiper.Spec.Conv1D

open Kuiper
open Kuiper.EMatrix
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality
module Seq = FStar.Seq

(* Implementation of the Conv1D functional spec.  Mirrors
   [Kuiper.Spec.Conv2D]: a primitive recursive accumulator
   [__conv1d_single] over the linearised (ic, k) prefix, plus
   the obvious extensionality machinery for the 3D erased tensor
   type. *)

let t3acc_pat (#et:Type) (#d0 #d1 #d2 : nat)
  (t : etensor3 et d0 d1 d2)
  (i : natlt d0) (j : natlt d1) (k : natlt d2)
  : Lemma (t3acc t i j k == t.f (i, j, k))
          [SMTPat (t.f (i, j, k))]
  = ()

let t3_equal #et #d0 #d1 #d2
  (t1 t2 : etensor3 et d0 d1 d2) : prop
  = forall (i:natlt d0) (j:natlt d1) (k:natlt d2).
      t3acc t1 i j k == t3acc t2 i j k

let lemma_t3_equal_intro #et #d0 #d1 #d2
  (t1 t2 : etensor3 et d0 d1 d2)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2).
                      t3acc t1 i j k == t3acc t2 i j k)
          (ensures t3_equal t1 t2)
          [SMTPat (t3_equal t1 t2)]
  = ()

let etensor3_ext #et #d0 #d1 #d2
  (t1 t2 : etensor3 et d0 d1 d2)
  : Lemma (requires t3_equal t1 t2)
          (ensures t1 == t2)
          [SMTPat (t3_equal t1 t2)]
  = let T3 f1 = t1 in
    let T3 f2 = t2 in
    let aux (idx : natlt d0 & natlt d1 & natlt d2)
      : Lemma (f1 idx == f2 idx)
      = let (i, j, k) = idx in
        assert (t3acc t1 i j k == t3acc t2 i j k)
    in
    Classical.forall_intro aux;
    F.extensionality_g _ _ f1 f2

let rec __conv1d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (#cout : nat)
  (kk : pos)
  (stride : pos)
  (pad : nat)
  (dilation : pos)
  (#l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cout cin kk)
  (b : natlt b_n)
  (oc : natlt cout)
  (ol : natlt l_out)
  (to : nat{to <= (if cin = 0 then 0 else cin * kk)})
  : GTot et (decreases to)
  = if to = 0 then zero
    else (
      let i = to - 1 in
      assert (cin > 0);
      let ic : natlt cin = unrank1_ic cin kk i in
      let k_i : natlt kk = unrank1_k cin kk i in
      let l_idx : int = ol * stride + k_i * dilation - pad in
      add
        (__conv1d_single kk stride pad dilation x weight b oc ol (to - 1))
        (mul (read_padded1 x b ic l_idx)
             (t3acc weight oc ic k_i))
    )

let __conv1d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (#cout : nat)
  (kk : pos)
  (stride : pos)
  (pad : nat)
  (dilation : pos)
  (#l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cout cin kk)
  (b : natlt b_n)
  (oc : natlt cout)
  (ol : natlt l_out)
  : Lemma
    (ensures __conv1d_single kk stride pad dilation x weight b oc ol 0 == zero)
    [SMTPat (__conv1d_single kk stride pad dilation x weight b oc ol 0)]
  = ()

let __conv1d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n : nat) (cin : pos) (#l_in : nat)
  (#cout : nat)
  (kk : pos)
  (stride : pos)
  (pad : nat)
  (dilation : pos)
  (#l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cout cin kk)
  (b : natlt b_n)
  (oc : natlt cout)
  (ol : natlt l_out)
  (to : pos{to <= cin * kk})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank1_ic cin kk i in
      let k_i = unrank1_k cin kk i in
      let l_idx : int = ol * stride + k_i * dilation - pad in
      __conv1d_single kk stride pad dilation x weight b oc ol to ==
      add
        (__conv1d_single kk stride pad dilation x weight b oc ol (to - 1))
        (mul (read_padded1 x b ic l_idx)
             (t3acc weight oc ic k_i))))
  = ()

let lemma_conv1d_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (#cout : nat)
  (kk : pos)
  (stride : pos)
  (pad : nat)
  (dilation : pos)
  (l_out : nat)
  (x : etensor3 et b_n cin l_in)
  (weight : etensor3 et cout cin kk)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout) (ol : natlt l_out)
  : Lemma (t3acc (conv1d kk stride pad dilation l_out x weight bias) b oc ol
           == conv1d_single kk stride pad dilation x weight bias b oc ol)
          [SMTPat (t3acc (conv1d kk stride pad dilation l_out x weight bias)
                         b oc ol)]
  = ()
