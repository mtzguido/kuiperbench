module Kuiper.Spec.Conv3D

(* Functional specification for standard 3D convolution
   (KernelBench L1 #54, #59, #60, #66 -- nn.Conv3d):

       y[b, oc, od, oh, ow] =
           bias[oc]
         + Σ_{ic, kd, kh, kw}
               x[b, ic,
                 od*Sd + kd*Dd - Pd,
                 oh*Sh + kh*Dh - Ph,
                 ow*Sw + kw*Dw - Pw]
             * w[oc, ic, kd, kh, kw]

   where:
     - input  x : (B, C_in,  D_in,  H_in,  W_in)
     - weight w : (C_out, C_in, K_d, K_h, K_w)
     - bias     : lseq et C_out
     - per-axis stride (Sd,Sh,Sw), padding (Pd,Ph,Pw),
       dilation (Dd,Dh,Dw), groups = 1
     - out-of-range input reads return [zero] (zero padding).

   Output spatial dims (d_out, h_out, w_out) are parameters; the
   caller is responsible for the standard relation
       d_out = (D_in + 2*Pd - Dd*(K_d-1) - 1) / Sd + 1   (etc.)

   Scalar-polymorphic; the partial sum is folded left-to-right
   over a linearised (ic, kd, kh, kw) index, fixing one specific
   summation order. *)

open Kuiper
open Kuiper.EMatrix
open FStar.FunctionalExtensionality { (^->>) }
module F = FStar.FunctionalExtensionality
module Seq = FStar.Seq

(* ------------------------------------------------------------------ *)
(* 5D erased tensor.                                                  *)
(* ------------------------------------------------------------------ *)

[@@erasable]
noeq
type etensor5 (et:Type) (d0 d1 d2 d3 d4 : nat) =
  | T5 : f:(natlt d0 & natlt d1 & natlt d2 & natlt d3 & natlt d4 ^->> et)
      -> etensor5 et d0 d1 d2 d3 d4

let mkT5 (#et:Type) (#d0 #d1 #d2 #d3 #d4 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> natlt d4 -> GTot et)
  : etensor5 et d0 d1 d2 d3 d4
  = T5 <| F.on_g _ <| fun (i, j, k, l, m) -> f i j k l m

let t5acc (#et:Type) (#d0 #d1 #d2 #d3 #d4 : nat)
  (t : etensor5 et d0 d1 d2 d3 d4)
  (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3) (m : natlt d4)
  : GTot et
  = t.f (i, j, k, l, m)

val t5acc_pat (#et:Type) (#d0 #d1 #d2 #d3 #d4 : nat)
  (t : etensor5 et d0 d1 d2 d3 d4)
  (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3) (m : natlt d4)
  : Lemma (t5acc t i j k l m == t.f (i, j, k, l, m))
          [SMTPat (t.f (i, j, k, l, m))]

val t5_equal (#et #d0 #d1 #d2 #d3 #d4 : _)
  (t1 t2 : etensor5 et d0 d1 d2 d3 d4) : prop

val lemma_t5_equal_intro (#et #d0 #d1 #d2 #d3 #d4 : _)
  (t1 t2 : etensor5 et d0 d1 d2 d3 d4)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2)
                           (l:natlt d3) (m:natlt d4).
                      t5acc t1 i j k l m == t5acc t2 i j k l m)
          (ensures t5_equal t1 t2)
          [SMTPat (t5_equal t1 t2)]

val etensor5_ext #et #d0 #d1 #d2 #d3 #d4
  (t1 t2 : etensor5 et d0 d1 d2 d3 d4)
  : Lemma (requires t5_equal t1 t2)
          (ensures t1 == t2)
          [SMTPat (t5_equal t1 t2)]

(* ------------------------------------------------------------------ *)
(* Padded 3-D read.                                                   *)
(* ------------------------------------------------------------------ *)

let read_padded3
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (b : natlt b_n)
  (ic : natlt cin)
  (d : int) (h : int) (w : int)
  : GTot et
  = if 0 <= d && d < d_in &&
       0 <= h && h < h_in &&
       0 <= w && w < w_in
    then t5acc x b ic d h w
    else zero

(* ------------------------------------------------------------------ *)
(* Linearised (ic, kd, kh, kw) index decomposition.                   *)
(* ------------------------------------------------------------------ *)

let unrank3_ic
  (cin : pos) (kd : pos) (kh : pos) (kw : pos)
  (i : nat{i < cin * kd * kh * kw})
  : Tot (natlt cin) =
  let d = kd * kh * kw in
  FStar.Math.Lemmas.lemma_div_le i (cin * d - 1) d;
  FStar.Math.Lemmas.cancel_mul_div cin d;
  FStar.Math.Lemmas.lemma_div_le 0 i d;
  FStar.Math.Lemmas.lemma_div_plus (-1) cin d;
  i / d

let unrank3_kd
  (cin : pos) (kd : pos) (kh : pos) (kw : pos)
  (i : nat{i < cin * kd * kh * kw})
  : Tot (natlt kd) =
  let d = kd * kh * kw in
  let r = i % d in
  FStar.Math.Lemmas.lemma_mod_lt i d;
  // r < kd*kh*kw, so r / (kh*kw) < kd
  FStar.Math.Lemmas.lemma_div_le r (kd * (kh * kw) - 1) (kh * kw);
  FStar.Math.Lemmas.cancel_mul_div kd (kh * kw);
  FStar.Math.Lemmas.lemma_div_plus (-1) kd (kh * kw);
  assert (kd * (kh * kw) == d);
  r / (kh * kw)

let unrank3_kh
  (cin : pos) (kd : pos) (kh : pos) (kw : pos)
  (i : nat{i < cin * kd * kh * kw})
  : Tot (natlt kh) =
  let r = (i % (kd * kh * kw)) % (kh * kw) in
  FStar.Math.Lemmas.lemma_mod_lt (i % (kd * kh * kw)) (kh * kw);
  FStar.Math.Lemmas.lemma_div_le r (kh * kw - 1) kw;
  FStar.Math.Lemmas.cancel_mul_div kh kw;
  FStar.Math.Lemmas.lemma_div_plus (-1) kh kw;
  r / kw

let unrank3_kw
  (cin : pos) (kd : pos) (kh : pos) (kw : pos)
  (i : nat{i < cin * kd * kh * kw})
  : Tot (natlt kw) =
  ((i % (kd * kh * kw)) % (kh * kw)) % kw

(* ------------------------------------------------------------------ *)
(* Per-output-pixel partial sum.                                      *)
(* ------------------------------------------------------------------ *)

val __conv3d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (b : natlt b_n)
  (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  (to : nat{to <= (if cin = 0 then 0 else cin * kd * kh * kw)})
  : GTot et

val __conv3d_single_zero_lemma
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  : Lemma
    (ensures __conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                             x weight b oc od oh ow 0 == zero)
    [SMTPat (__conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                             x weight b oc od oh ow 0)]

val __conv3d_single_lemma
  (#et:Type) {| scalar et |}
  (#b_n : nat) (cin : pos) (#d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  (to : pos{to <= cin * kd * kh * kw})
  : Lemma
    (ensures (
      let i = to - 1 in
      let ic = unrank3_ic cin kd kh kw i in
      let kd_i = unrank3_kd cin kd kh kw i in
      let kh_i = unrank3_kh cin kd kh kw i in
      let kw_i = unrank3_kw cin kd kh kw i in
      let d_idx : int = od * sd + kd_i * dd - pd in
      let h_idx : int = oh * sh + kh_i * dh - ph in
      let w_idx : int = ow * sw + kw_i * dw - pw in
      __conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                      x weight b oc od oh ow to ==
      add
        (__conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                         x weight b oc od oh ow (to - 1))
        (mul (read_padded3 x b ic d_idx h_idx w_idx)
             (t5acc weight oc ic kd_i kh_i kw_i))))

(* ------------------------------------------------------------------ *)
(* Full per-pixel result and full output tensor.                      *)
(* ------------------------------------------------------------------ *)

let conv3d_single
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (#d_out #h_out #w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  : GTot et
  = let to : nat = if cin = 0 then 0 else cin * kd * kh * kw in
    add (Seq.index bias oc)
        (__conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                         x weight b oc od oh ow to)

let conv3d
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (d_out h_out w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (bias : Seq.lseq et cout)
  : etensor5 et b_n cout d_out h_out w_out
  = mkT5 (fun b oc od oh ow ->
      conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                    x weight bias b oc od oh ow)

val lemma_conv3d_index
  (#et:Type) {| scalar et |}
  (#b_n #cin #d_in #h_in #w_in : nat)
  (#cout : nat)
  (kd : pos) (kh : pos) (kw : pos)
  (sd : pos) (sh : pos) (sw : pos)
  (pd : nat) (ph : nat) (pw : nat)
  (dd : pos) (dh : pos) (dw : pos)
  (d_out h_out w_out : nat)
  (x : etensor5 et b_n cin d_in h_in w_in)
  (weight : etensor5 et cout cin kd kh kw)
  (bias : Seq.lseq et cout)
  (b : natlt b_n) (oc : natlt cout)
  (od : natlt d_out) (oh : natlt h_out) (ow : natlt w_out)
  : Lemma (t5acc (conv3d kd kh kw sd sh sw pd ph pw dd dh dw
                         d_out h_out w_out x weight bias)
                 b oc od oh ow
           == conv3d_single kd kh kw sd sh sw pd ph pw dd dh dw
                            x weight bias b oc od oh ow)
          [SMTPat (t5acc (conv3d kd kh kw sd sh sw pd ph pw dd dh dw
                                 d_out h_out w_out x weight bias)
                         b oc od oh ow)]
