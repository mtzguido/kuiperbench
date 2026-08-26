module Kuiper.Spec.Conv1D

(* Functional specification for standard 1D convolution
   (KernelBench L1 #67, #76 -- nn.Conv1d):

       y[b, oc, ol] = bias[oc]
                    + Σ_{ic, k}
                          x[b, ic, ol*S + k*D - P]
                          * w[oc, ic, k]

   where:
     - input  x : (B, C_in,  L_in)
     - weight w : (C_out, C_in, K)
     - bias     : lseq et C_out
     - stride S, padding P, dilation D, groups = 1
     - out-of-range input reads return [zero] (zero padding).

   Output spatial dimension is a parameter [l_out] and the caller
   must satisfy
       l_out = (L_in + 2*P - D*(K-1) - 1) / S + 1.

   The spec is [scalar et]-polymorphic.  Like [Kuiper.Spec.Conv2D]
   the partial sum is folded left-to-right over a linearised
   (ic, k) index, fixing one specific summation order; floating
   point implementations differing in associativity must use a
   decomposition lemma analogous to [matmul_tiles_lemma]. *)

open Kuiper
open Kuiper.EMatrix
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* 3D erased tensor.                                                  *)
(* ------------------------------------------------------------------ *)

[@@erasable]
noeq
type etensor3 (et:Type) (d0 d1 d2 : nat) =
  | T3 : f:(natlt d0 & natlt d1 & natlt d2 ^->> et)
      -> etensor3 et d0 d1 d2

let mkT3 (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  : etensor3 et d0 d1 d2
  = T3 <| F.on_g _ <| fun (i, j, k) -> f i j k

let t3acc (#et:Type) (#d0 #d1 #d2 : nat)
  (t : etensor3 et d0 d1 d2)
  (i : natlt d0) (j : natlt d1) (k : natlt d2)
  : GTot et
  = t.f (i, j, k)

val t3acc_pat (#et:Type) (#d0 #d1 #d2 : nat)
  (t : etensor3 et d0 d1 d2)
  (i : natlt d0) (j : natlt d1) (k : natlt d2)
  : Lemma (t3acc t i j k == t.f (i, j, k))
          [SMTPat (t.f (i, j, k))]

val t3_equal (#et #d0 #d1 #d2 : _)
  (t1 t2 : etensor3 et d0 d1 d2) : prop

val lemma_t3_equal_intro (#et #d0 #d1 #d2 : _)
  (t1 t2 : etensor3 et d0 d1 d2)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2).
                      t3acc t1 i j k == t3acc t2 i j k)
          (ensures t3_equal t1 t2)
          [SMTPat (t3_equal t1 t2)]

val etensor3_ext #et #d0 #d1 #d2
  (t1 t2 : etensor3 et d0 d1 d2)
  : Lemma (requires t3_equal t1 t2)
          (ensures t1 == t2)
          [SMTPat (t3_equal t1 t2)]

(* ------------------------------------------------------------------ *)
(* Padded 1-D read.                                                   *)
(* ------------------------------------------------------------------ *)

let read_padded1
  (#et:Type) {| scalar et |}
  (#b_n #cin #l_in : nat)
  (x : etensor3 et b_n cin l_in)
  (b : natlt b_n)
  (ic : natlt cin)
  (l : int)
  : GTot et
  = if 0 <= l && l < l_in
    then t3acc x b ic l
    else zero

(* ------------------------------------------------------------------ *)
(* Linearised (ic, k) index decomposition.                            *)
(* ------------------------------------------------------------------ *)

let unrank1_ic
  (cin : pos) (kk : pos)
  (i : nat{i < cin * kk})
  : Tot (natlt cin) =
  FStar.Math.Lemmas.lemma_div_le i (cin * kk - 1) kk;
  FStar.Math.Lemmas.cancel_mul_div cin kk;
  FStar.Math.Lemmas.lemma_div_le 0 i kk;
  FStar.Math.Lemmas.lemma_div_plus (-1) cin kk;
  i / kk

let unrank1_k
  (cin : pos) (kk : pos)
  (i : nat{i < cin * kk})
  : Tot (natlt kk) =
  i % kk

(* ------------------------------------------------------------------ *)
(* Per-output partial sum.                                            *)
(* ------------------------------------------------------------------ *)

val __conv1d_single
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
  : GTot et

val __conv1d_single_zero_lemma
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

val __conv1d_single_lemma
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

(* ------------------------------------------------------------------ *)
(* Full per-output result and full output tensor.                     *)
(* ------------------------------------------------------------------ *)

let conv1d_single
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
  (bias : Seq.lseq et cout)
  (b : natlt b_n)
  (oc : natlt cout)
  (ol : natlt l_out)
  : GTot et
  = let to : nat = if cin = 0 then 0 else cin * kk in
    add (Seq.index bias oc)
        (__conv1d_single kk stride pad dilation x weight b oc ol to)

let conv1d
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
  : etensor3 et b_n cout l_out
  = mkT3 (fun b oc ol ->
      conv1d_single kk stride pad dilation x weight bias b oc ol)

val lemma_conv1d_index
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
