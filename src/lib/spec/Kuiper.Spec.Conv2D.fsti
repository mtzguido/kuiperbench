module Kuiper.Spec.Conv2D

(* Functional specification for standard 2D convolution
   (KernelBench L1 #50 -- nn.Conv2d):

       y[b, oc, oh, ow] = bias[oc]
                        + Σ_{ic, kh, kw}
                              x[b, ic, oh*S + kh - P, ow*S + kw - P]
                              * w[oc, ic, kh, kw]

   where:
     - input  x : (B, C_in,  H_in,  W_in)
     - weight w : (C_out, C_in, K_h, K_w)
     - bias     : lseq et C_out
     - stride S, padding P, dilation = 1, groups = 1
     - out-of-range input reads return [zero] (zero padding).

   The output spatial dimensions are not computed inside the spec --
   they are taken as parameters [h_out], [w_out] and the caller is
   responsible for the standard relation
       h_out = (H_in + 2*P - K_h) / S + 1
       w_out = (W_in + 2*P - K_w) / S + 1
   (with dilation = 1).  This avoids forcing divisibility constraints
   into the type signature.

   The spec is defined at the [scalar et] level using the canonical
   [add]/[mul]/[zero] of the typeclass.  The accumulation is folded
   left-to-right over a linearised (ic, kh, kw) index, matching the
   convention used in [Kuiper.Spec.GEMM.__matmul_single]: a
   floating-point implementation is allowed to differ in associativity
   and only needs to match the result up to [%~] (real-valued
   approximation).  The exact, scalar-level spec given here is
   strong enough to recover an approximate spec via
   [Kuiper.EMatrix.lemma_to_real_matrix_approximates] and the
   per-step [a_add]/[a_mul] lemmas of [real_like]. *)

open Kuiper
open Kuiper.EMatrix
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* 4D erased tensor, analogous to [chest2].                          *)
(* ------------------------------------------------------------------ *)

[@@erasable]
noeq
type etensor4 (et:Type) (d0 d1 d2 d3 : nat) =
  | T4 : f:(natlt d0 & natlt d1 & natlt d2 & natlt d3 ^->> et)
      -> etensor4 et d0 d1 d2 d3

let mkT4 (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> GTot et)
  : etensor4 et d0 d1 d2 d3
  = T4 <| F.on_g _ <| fun (i, j, k, l) -> f i j k l

let tacc (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (t : etensor4 et d0 d1 d2 d3)
  (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3)
  : GTot et
  = t.f (i, j, k, l)

val tacc_pat (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (t : etensor4 et d0 d1 d2 d3)
  (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3)
  : Lemma (tacc t i j k l == t.f (i, j, k, l))
          [SMTPat (t.f (i, j, k, l))]

val t4_equal (#et #d0 #d1 #d2 #d3 : _)
  (t1 t2 : etensor4 et d0 d1 d2 d3) : prop

val lemma_t4_equal_intro (#et #d0 #d1 #d2 #d3 : _)
  (t1 t2 : etensor4 et d0 d1 d2 d3)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3).
                      tacc t1 i j k l == tacc t2 i j k l)
          (ensures t4_equal t1 t2)
          [SMTPat (t4_equal t1 t2)]

val etensor4_ext #et #d0 #d1 #d2 #d3
  (t1 t2 : etensor4 et d0 d1 d2 d3)
  : Lemma (requires t4_equal t1 t2)
          (ensures t1 == t2)
          [SMTPat (t4_equal t1 t2)]

(* ------------------------------------------------------------------ *)
(* Padded read: out-of-range spatial indices return [zero].           *)
(* ------------------------------------------------------------------ *)

let read_padded
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (b : natlt b_n)
  (ic : natlt cin)
  (h : int)
  (w : int)
  : GTot et
  = if 0 <= h && h < h_in && 0 <= w && w < w_in
    then tacc x b ic h w
    else zero

(* ------------------------------------------------------------------ *)
(* Linearised (ic, kh, kw) index decomposition.                       *)
(* ------------------------------------------------------------------ *)

(* Decompose a flat index [i < cin*kh*kw] into [(ic, h, w)].          *)
let unrank_ic
  (cin : pos) (kh : pos) (kw : pos)
  (i : nat{i < cin * kh * kw})
  : Tot (natlt cin) =
  let d = kh * kw in
  FStar.Math.Lemmas.lemma_div_le i (cin * d - 1) d;
  FStar.Math.Lemmas.cancel_mul_div cin d;
  FStar.Math.Lemmas.lemma_div_le 0 i d;
  // (cin*d - 1)/d == cin - 1 since (cin-1)*d <= cin*d-1 < cin*d.
  FStar.Math.Lemmas.lemma_div_plus (-1) cin d;
  i / d

let unrank_kh
  (cin : pos) (kh : pos) (kw : pos)
  (i : nat{i < cin * kh * kw})
  : Tot (natlt kh) =
  let r = i % (kh * kw) in
  FStar.Math.Lemmas.lemma_mod_lt i (kh * kw);
  // r < kh*kw, so r/kw < kh
  FStar.Math.Lemmas.lemma_div_le r (kh * kw - 1) kw;
  FStar.Math.Lemmas.cancel_mul_div kh kw;
  FStar.Math.Lemmas.lemma_div_plus (-1) kh kw;
  r / kw

let unrank_kw
  (cin : pos) (kh : pos) (kw : pos)
  (i : nat{i < cin * kh * kw})
  : Tot (natlt kw) =
  (i % (kh * kw)) % kw

(* ------------------------------------------------------------------ *)
(* Per-output-pixel partial sum, folded over a prefix of the          *)
(* linearised (ic, kh, kw) index.  Mirrors                            *)
(* [Kuiper.Spec.GEMM.__matmul_single].                                *)
(* ------------------------------------------------------------------ *)

val __conv2d_single
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
  : GTot et

val __conv2d_single_zero_lemma
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

val __conv2d_single_lemma
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

(* ------------------------------------------------------------------ *)
(* Full per-pixel result and full output tensor.                      *)
(* ------------------------------------------------------------------ *)

let conv2d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #h_in #w_in : nat)
  (#cout : nat)
  (kh : pos) (kw : pos)
  (stride : pos)
  (pad : nat)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n cin h_in w_in)
  (weight : etensor4 et cout cin kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n)
  (oc : natlt cout)
  (oh : natlt h_out)
  (ow : natlt w_out)
  : GTot et
  = let to : nat = if cin = 0 then 0 else cin * kh * kw in
    add (Seq.index bias oc)
        (__conv2d_single kh kw stride pad x weight b oc oh ow to)

let conv2d
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
  : etensor4 et b_n cout h_out w_out
  = mkT4 (fun b oc oh ow ->
      conv2d_single kh kw stride pad x weight bias b oc oh ow)

val lemma_conv2d_index
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
