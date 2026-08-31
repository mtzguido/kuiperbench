module Kuiper.KB.SDPA

(* Implementation: see Kuiper.KB.SDPA.fsti for the design overview.

   KernelBench L1 #97 — Scaled Dot-Product Attention.

     out = softmax((Q @ K^T) / sqrt(D)) @ V

   This module is a verified ORCHESTRATOR: it chains four already-verified
   Kuiper primitives over the SAME GPU buffers (no fused kernel, no data
   movement beyond what each primitive performs):

     1. batched_gemm_f32 bh s d s gQ gKT gScores  (gScores := Q @ K^T, exact)
     2. ghost-reshape (bh,s,s) Array3 -> (bh*s*s) Array1, scalar-multiply by
        [scale] in place via smul_fw_f32, ghost-reshape back     (exact)
     3. ghost-reshape (bh,s,s) Array3 -> (bh*s, s) Array2, row-softmax along
        the last dim via Klas.RowSoftmax.row_softmax_rm_f32, reshape back
                                                                (the %~ step)
     4. batched_gemm_f32 bh s s d gScores gV gOut  (gOut := probs @ V, exact)

   The ghost reshapes are pure re-interpretations of the SAME buffer; they are
   the analogues of the Array3<->Array2 reshapes in Kuiper.KB.MatmulND, plus a
   new Array3<->Array1 reshape for the scalar-multiply.

   The direct real scale proof uses the temporary [rsqrt_approx]
   compatibility assumption documented in the repository patch. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.EMatrix
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module RS = Kuiper.Kernel.RowSoftmax
module RSI = Klas.RowSoftmax
module SMul = Kuiper.KB.ScalarMul
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module SqrtApprox = Kuiper.KB.Compat.SqrtApprox
module RealSqrt = FStar.Math.Sqrt
open Kuiper.KB.BatchedGEMM { batched_matmul, batched_gemm_f32 }
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Bijection

(* Verified attention scale, inlined into the public entry point. *)
inline_for_extraction noextract
let sdpa_scale_f32 (d : szp) : f32 =
  rsqrt (of_int (FStar.Int.Cast.uint64_to_int64
                   (FStar.SizeT.sizet_to_uint64 d)))

(* ----------------------------------------------------------------------- *)
(* Reshape glue: Array3 (n,m,k) <-> Array2 (n*m,k).  Direct copies of the   *)
(* lemmas in Kuiper.KB.MatmulND, generalized to take the flattened row      *)
(* dimension [p] explicitly (so it can be a [SZ.v] product matching the     *)
(* kernel's szp argument).                                                  *)
(* ----------------------------------------------------------------------- *)

let major_on_zero_f
  (#nn:nat) (kk:nat) (#d:shape nn) (sub:layout_f_for d)
  (a:natlt kk) (rest:abs d)
  : Lemma ((major_on 0 kk sub).f (a, rest) == a * sizeof d + sub.f rest)
  = assert ((major_on 0 kk sub).f (a, rest)
             == major_on_f 0 kk sub (a, rest));
    assert_norm (major_on_f #nn 0 kk #d sub (a, rest)
                   == a * sizeof d + sub.f rest)

let imap_eq
  (#n #m #k : nat)
  (i:nat{i<n}) (j:nat{j<m}) (c:nat{c<k})
  : Lemma
    ((l3_batched_row_major n m k).imap.f (i,(j,(c,())))
       == (l2_row_major (n*m) k).imap.f ((i*m+j),(c,())))
  = let l_k : layout_f_for (k @| INil) = major_on 0 k lunit in
    let l_mk : layout_f_for (m @| k @| INil) = major_on 0 m l_k in
    FStar.Math.Lemmas.lemma_mult_le_right m (i+1) n;
    FStar.Math.Lemmas.distributivity_add_left i 1 m;
    let a2 : natlt (n*m) = i*m + j in
    assert_norm (sizeof (k @| INil) == k * 1);
    assert (sizeof (m @| k @| INil) == m * k);
    assert (lunit.f () == 0);
    assert ((l3_batched_row_major n m k).imap.f (i,(j,(c,())))
              == (major_on 0 n l_mk).f (i,(j,(c,()))));
    assert ((l2_row_major (n*m) k).imap.f (a2,(c,()))
              == (major_on 0 (n*m) l_k).f (a2,(c,())));
    major_on_zero_f #2 n #(m @| k @| INil) l_mk i (j,(c,()));
    major_on_zero_f #1 m #(k @| INil) l_k j (c,());
    major_on_zero_f #0 k #INil lunit c ();
    major_on_zero_f #1 (n*m) #(k @| INil) l_k a2 (c,());
    assert (sizeof INil == 1);
    FStar.Math.Lemmas.paren_mul_right i m k;
    FStar.Math.Lemmas.distributivity_add_left (i*m) j k

(* [imap_eq] specialized to an explicit flat dimension [p == n*m]. *)
let imap_eq_p
  (#n #m #k : nat) (p:nat) (_:squash (p == n * m))
  (i:nat{i<n}) (j:nat{j<m}) (c:nat{c<k})
  : Lemma
    ((l3_batched_row_major n m k).imap.f (i,(j,(c,())))
       == (l2_row_major p k).imap.f ((i*m+j),(c,())))
  = imap_eq #n #m #k i j c

let content_ok
  (#et:Type) (#n #m #k : nat) (p:nat) (_:squash (p == n * m))
  (s3 : chest3 et n m k)
  (i:nat{i<n}) (j:nat{j<m}) (c:nat{c<k})
  : Lemma
    (acc2
       (from_seq (l2_row_major p k)
          (to_seq (l3_batched_row_major n m k) s3))
       (i*m+j) c
     == acc3 s3 i j c)
  = imap_eq_p #n #m #k p () i j c;
    let l3 = l3_batched_row_major n m k in
    let l2 = l2_row_major p k in
    let s_flat = to_seq l3 s3 in
    assert (acc2 (from_seq l2 s_flat) (i*m+j) c
              == Seq.index s_flat (l2.imap.f ((i*m+j),(c,()))));
    assert (l2.imap.f ((i*m+j),(c,())) == l3.imap.f (i,(j,(c,()))));
    assert (Kuiper.Injection.inverse_f l3.imap (l3.imap.f (i,(j,(c,())))) == (i,(j,(c,()))))

(* [from_seq (to_seq s) == s] for Array2 (the round-trip the library does not
   provide directly).  Mirror of MatmulND.from_to2. *)
let from_to2
  (#et:Type) (#m #n:nat)
  (l : full_layout2 m n)
  (s : chest2 et m n)
  : Lemma (ensures from_seq l (to_seq l s) == s)
  = let lhs = from_seq l (to_seq l s) in
    let aux (i:natlt m) (j:natlt n)
      : Lemma (acc2 lhs i j == acc2 s i j)
      = inverse_lem l.imap (l.imap.f (i,(j,())));
        l.imap.is_inj (inverse_f l.imap (l.imap.f (i,(j,())))) (i,(j,()))
    in
    Classical.forall_intro_2 aux;
    EM.lemma_equal_intro lhs s;
    Kuiper.Chest.ext lhs s

ghost
fn reshape3to2
  (#et:Type)
  (#n #m #k : nat)
  (p : nat)
  (#_ : squash (p == n * m))
  (a3 : array3 et (l3_batched_row_major n m k))
  (#s3 : chest3 et n m k)
  (#f : perm)
  requires
    a3 |-> Frac f s3
  ensures
    from_array (l2_row_major p k) (core a3)
      |-> Frac f (from_seq (l2_row_major p k)
                     (to_seq (l3_batched_row_major n m k) s3))
{
  tensor_concr a3;
  tensor_abs' (l2_row_major p k) (core a3)
}

ghost
fn reshape2to3
  (#et:Type)
  (#n #m #k : nat)
  (p : nat)
  (#_ : squash (p == n * m))
  (a3 : array3 et (l3_batched_row_major n m k))
  (#s3 : chest3 et n m k)
  (#f : perm)
  requires
    from_array (l2_row_major p k) (core a3)
      |-> Frac f (from_seq (l2_row_major p k)
                     (to_seq (l3_batched_row_major n m k) s3))
  ensures
    a3 |-> Frac f s3
{
  tensor_concr (from_array (l2_row_major p k) (core a3));
  rewrite
    (core (from_array (l2_row_major p k) (core a3))
      |-> Frac f (to_seq (l2_row_major p k)
                    (from_seq (l2_row_major p k)
                       (to_seq (l3_batched_row_major n m k) s3))))
  as
    (core a3 |-> Frac f (to_seq (l3_batched_row_major n m k) s3));
  tensor_abs (l3_batched_row_major n m k) (core a3) #f #s3;
  rewrite
    (from_array (l3_batched_row_major n m k) (core a3) |-> Frac f s3)
  as
    (a3 |-> Frac f s3);
}

ghost
fn reshape2to3_eq
  (#et:Type)
  (#n #m #k : nat)
  (p : nat)
  (#_ : squash (p == n * m))
  (a3 : array3 et (l3_batched_row_major n m k))
  (#s3 : chest3 et n m k)
  (#f : perm)
  (#e : chest2 et p k)
  (#_ : squash (
     e == from_seq (l2_row_major p k)
            (to_seq (l3_batched_row_major n m k) s3)))
  requires
    from_array (l2_row_major p k) (core a3) |-> Frac f e
  ensures
    a3 |-> Frac f s3
{
  rewrite
    (from_array (l2_row_major p k) (core a3) |-> Frac f e)
  as
    (from_array (l2_row_major p k) (core a3)
      |-> Frac f (from_seq (l2_row_major p k)
                     (to_seq (l3_batched_row_major n m k) s3)));
  reshape2to3 p a3 #s3 #f;
}

(* ----------------------------------------------------------------------- *)
(* Reshape glue: Array3 (n,m,k) <-> Array1 (n*m*k).  New here (no Array1    *)
(* reshape in MatmulND); same structure, with [l1_forward] in place of      *)
(* [l2_row_major].                                                          *)
(* ----------------------------------------------------------------------- *)

ghost
fn reshape3to1
  (#et:Type)
  (#n #m #k : nat)
  (p : nat)
  (#_ : squash (p == n * m * k))
  (a3 : array3 et (l3_batched_row_major n m k))
  (#s3 : chest3 et n m k)
  (#f : perm)
  requires
    a3 |-> Frac f s3
  ensures
    from_array (l1_forward p) (core a3)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l3_batched_row_major n m k) s3))
{
  tensor_concr a3;
  tensor_abs' (l1_forward p) (core a3)
}

ghost
fn reshape1to3
  (#et:Type)
  (#n #m #k : nat)
  (p : nat)
  (#_ : squash (p == n * m * k))
  (a3 : array3 et (l3_batched_row_major n m k))
  (#s3 : chest3 et n m k)
  (#f : perm)
  requires
    from_array (l1_forward p) (core a3)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l3_batched_row_major n m k) s3))
  ensures
    a3 |-> Frac f s3
{
  tensor_concr (from_array (l1_forward p) (core a3));
  rewrite
    (core (from_array (l1_forward p) (core a3))
      |-> Frac f (to_seq (l1_forward p)
                    (from_seq (l1_forward p)
                       (to_seq (l3_batched_row_major n m k) s3))))
  as
    (core a3 |-> Frac f (to_seq (l3_batched_row_major n m k) s3));
  tensor_abs (l3_batched_row_major n m k) (core a3) #f #s3;
  rewrite
    (from_array (l3_batched_row_major n m k) (core a3) |-> Frac f s3)
  as
    (a3 |-> Frac f s3);
}

ghost
fn reshape1to3_eq
  (#et:Type)
  (#n #m #k : nat)
  (p : nat)
  (#_ : squash (p == n * m * k))
  (a3 : array3 et (l3_batched_row_major n m k))
  (#s3 : chest3 et n m k)
  (#f : perm)
  (#e : chest1 et p)
  (#_ : squash (
     e == from_seq (l1_forward p)
            (to_seq (l3_batched_row_major n m k) s3)))
  requires
    from_array (l1_forward p) (core a3) |-> Frac f e
  ensures
    a3 |-> Frac f s3
{
  rewrite
    (from_array (l1_forward p) (core a3) |-> Frac f e)
  as
    (from_array (l1_forward p) (core a3)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l3_batched_row_major n m k) s3)));
  reshape1to3 p a3 #s3 #f;
}

(* ----------------------------------------------------------------------- *)
(* Pure spec lemmas.                                                        *)
(* ----------------------------------------------------------------------- *)

(* The in-place scalar-multiply over the flat Array1 view corresponds exactly
   to [mscale] on the Array3 tensor: mapping [mul c] over the reshaped flat
   sequence equals the reshaped flat of [mscale c s3].  Pure index reasoning
   (both sides are length-[p] sequences; [mul c] commutes with the layout
   permutation). *)
#push-options " --fuel 4 --ifuel 2"
let smul_reshape_eq
  (c : f32) (#n #m #k : nat) (p:nat) (_:squash (p == n * m * k))
  (s3 : chest3 f32 n m k)
  : Lemma
    (chest_map (mul c)
        (from_seq (l1_forward p)
           (to_seq (l3_batched_row_major n m k) s3))
     == from_seq (l1_forward p)
           (to_seq (l3_batched_row_major n m k) (mscale c s3)))
  = let l3 = l3_batched_row_major n m k in
    let l1 = l1_forward p in
    let lhs = chest_map (mul c)
                (from_seq l1 (to_seq l3 s3)) in
    let rhs = from_seq l1 (to_seq l3 (mscale c s3)) in
    introduce forall (i : abs (p @| INil)). acc lhs i == acc rhs i
    with (let (idx, ()) = i in
          let q = l1.imap.f (idx, ()) in
          ());
    Kuiper.Chest.lemma_equal_intro lhs rhs;
    Kuiper.Chest.ext lhs rhs
#pop-options

(* Row correspondence: row [p*s + i] of the flattened real scores equals row
   [i] of page [p] of the page-wise real scores.  Softmax is row-local, so the
   1-D softmax of these equal rows is equal.  Pure [Seq] extensionality plus
   [content_ok]. *)
let row_corr
  (#et:Type) (#bh #s : nat) (p:nat) (_:squash (p == bh * s))
  (scaled : chest3 et bh s s)
  (pp : natlt bh) (ii : natlt s)
  : Lemma
    (EM.ematrix_row
       (from_seq (l2_row_major p s)
          (to_seq (l3_batched_row_major bh s s) scaled))
       (pp * s + ii)
     == EM.ematrix_row
          (slice_page scaled pp)
          ii)
  = let flat = from_seq (l2_row_major p s)
                 (to_seq (l3_batched_row_major bh s s) scaled) in
    let lhs = EM.ematrix_row flat (pp * s + ii) in
    let rhs = EM.ematrix_row (slice_page scaled pp) ii in
    let aux (j:natlt s) : Lemma (Seq.index lhs j == Seq.index rhs j) =
      content_ok #et #bh #s #s p () scaled pp ii j
    in
    Classical.forall_intro aux;
    Seq.lemma_eq_intro lhs rhs

(* Cell of a chest2 row equals the matrix cell (needs fuel to compute through
   the [chest_slice] bijection). *)
#push-options "--fuel 6 --ifuel 4"
let acc1_chest2_row (#et:Type) (#r #cc:nat) (x:chest2 et r cc) (i:natlt r) (j:natlt cc)
  : Lemma (acc1 (chest2_row x i) j == acc2 x i j)
  = ()

let chest2_row_to_seq (#et:Type) (#r #cc:nat)
  (x:chest2 et r cc) (i:natlt r)
  : Lemma (chest1_to_seq (chest2_row x i) == EM.ematrix_row x i)
  = Classical.forall_intro (acc1_chest2_row x i);
    Seq.lemma_eq_intro (chest1_to_seq (chest2_row x i)) (EM.ematrix_row x i)

let index_chest1_to_seq (#et:Type) (#n:nat)
  (c:chest1 et n) (j:natlt n)
  : Lemma (Seq.index (chest1_to_seq c) j == acc1 c j)
  = ()

(* Lift [ematrix_row] (lseq) equality to [chest2_row] (chest1) equality; lets
   [row_corr]'s lseq statement feed the [chest2_row]-based [softmax_pages]. *)
let chest2_row_cong (#et:Type) (#r1 #r2 #cc:nat)
  (x : chest2 et r1 cc) (y : chest2 et r2 cc) (i : natlt r1) (i' : natlt r2)
  : Lemma (requires EM.ematrix_row x i == EM.ematrix_row y i')
          (ensures chest2_row x i == chest2_row y i')
  = introduce forall (t : abs (cc @| INil)). acc (chest2_row x i) t == acc (chest2_row y i') t
    with (let (jj, ()) = t in
          acc1_chest2_row x i jj;
          acc1_chest2_row y i' jj;
          chest2_row_to_seq x i;
          chest2_row_to_seq y i';
          assert (Seq.index (EM.ematrix_row x i) jj == Seq.index (EM.ematrix_row y i') jj);
          assert (Seq.index (chest1_to_seq (chest2_row x i)) jj ==
                  Seq.index (chest1_to_seq (chest2_row y i')) jj);
          index_chest1_to_seq (chest2_row x i) jj;
          index_chest1_to_seq (chest2_row y i') jj;
          assert (acc1 (chest2_row x i) jj == acc1 (chest2_row y i') jj);
          assert (acc (chest2_row x i) (jj, ()) == acc1 (chest2_row x i) jj);
          assert (acc (chest2_row y i') (jj, ()) == acc1 (chest2_row y i') jj));
    Kuiper.Chest.lemma_equal_intro (chest2_row x i) (chest2_row y i');
    Kuiper.Chest.ext (chest2_row x i) (chest2_row y i')
#pop-options

(* Bridge pointwise (flat 3-index) approximation to whole-chest3 [%~]; replaces
   the deleted [EMatrix3.lemma_approximates_intro]. *)
let lemma_approximates_intro3
  (#et:Type) {| scalar et, real_like et |} (#d0 #d1 #d2 : nat)
  (m1 : chest3 et d0 d1 d2) (m2 : chest3 real d0 d1 d2)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2).
                      acc3 m1 i j k %~ acc3 m2 i j k)
          (ensures m1 %~ m2)
  = introduce forall (idx : abs (d0 @| d1 @| d2 @| INil)). acc m1 idx %~ acc m2 idx
    with (let (i, (j, (k, ()))) = idx in
          assert (acc3 m1 i j k %~ acc3 m2 i j k))

let sdpa_scale_approx (d : szp)
  : Lemma (sdpa_scale_f32 d %~ real_sdpa_scale d)
  = let d64 : Int64.t = FStar.Int.Cast.uint64_to_int64
      (FStar.SizeT.sizet_to_uint64 d) in
    assert (Int64.v d64 == SZ.v d);
    let df : f32 = of_int d64 in
    of_int_approx #f32 d64;
    assert (df %~ FStar.Real.of_int d);
    SqrtApprox.rsqrt_approx df (FStar.Real.of_int d)

let batched_matmul_approx
  (#batch #rows #shared #cols:nat)
  (eA : chest3 f32 batch rows shared)
  (eB : chest3 f32 batch shared cols)
  (rA : chest3 real batch rows shared)
  (rB : chest3 real batch shared cols)
  : Lemma
      (requires eA %~ rA /\ eB %~ rB)
      (ensures batched_matmul eA eB %~ batched_matmul rA rB)
  = let eC : chest3 f32 batch rows cols = mk3 (fun _ _ _ -> zero) in
    let rC : chest3 real batch rows cols = to_real_chest eC in
    lemma_to_real_chest_approximates eC;
    assert (approx2 (MS.comb2 #f32) (MS.comb2 #real));
    MU.bmmcomb_approx_real (MS.comb2 #f32) (MS.comb2 #real)
      eA eB eC rA rB rC;
    MS.bmatmul_is_bgemm eC eA eB;
    MS.bmatmul_is_bgemm rC rA rB

let mscale_approx
  (#d0 #d1 #d2:nat)
  (c : f32) (rc : real)
  (m : chest3 f32 d0 d1 d2) (rm : chest3 real d0 d1 d2)
  : Lemma
      (requires c %~ rc /\ m %~ rm)
      (ensures mscale c m %~ mscale rc rm)
  = let aux (i:natlt d0) (j:natlt d1) (k:natlt d2)
      : Lemma (acc3 (mscale c m) i j k %~ acc3 (mscale rc rm) i j k)
      = a_mul c (acc3 m i j k) rc (acc3 rm i j k)
    in
    Classical.forall_intro_3 aux;
    lemma_approximates_intro3 (mscale c m) (mscale rc rm)

let reshape3to2_approx
  (#n #m #k:nat) (p:nat) (_:squash (p == n * m))
  (e : chest3 f32 n m k) (r : chest3 real n m k)
  : Lemma
      (requires e %~ r)
      (ensures
        from_seq (l2_row_major p k) (to_seq (l3_batched_row_major n m k) e)
        %~
        from_seq (l2_row_major p k) (to_seq (l3_batched_row_major n m k) r))
  = let e2 = from_seq (l2_row_major p k)
      (to_seq (l3_batched_row_major n m k) e) in
    let r2 = from_seq (l2_row_major p k)
      (to_seq (l3_batched_row_major n m k) r) in
    let aux (q:natlt p) (j:natlt k)
      : Lemma (acc2 e2 q j %~ acc2 r2 q j)
      = let i : natlt n = q / m in
        let ii : natlt m = q % m in
        content_ok #f32 #n #m #k p () e i ii j;
        content_ok #real #n #m #k p () r i ii j
    in
    Classical.forall_intro_2 aux;
    EM.lemma_approximates_intro e2 r2

(* The softmax correspondence: if [sa'] (the Array2 result) approximates the
   per-row real softmax of a flattened real tensor, and [probs] re-interprets [sa'] as
   the (bh,s,s) tensor (flat3to2 probs == sa'), then [probs] approximates the
   page-wise real softmax of [scaled]. *)
#push-options "--fuel 6 --ifuel 4"
let softmax_corr
  (#bh : nat) (#s : nat{s > 0}) (p:nat) (_:squash (p == bh * s))
  (rscaled : chest3 real bh s s)
  (sa' : chest2 f32 p s)
  (probs : chest3 f32 bh s s)
  (_ : squash (
     sa' %~ RS.row_softmax_real
              (from_seq (l2_row_major p s)
                 (to_seq (l3_batched_row_major bh s s) rscaled))))
  (_ : squash (
     from_seq (l2_row_major p s)
        (to_seq (l3_batched_row_major bh s s) probs) == sa'))
  : Lemma
    (probs %~
       (softmax_pages rscaled))
  = let l2 = l2_row_major p s in
    let l3 = l3_batched_row_major bh s s in
    let ra = from_seq l2 (to_seq l3 rscaled) in
    let rm = rscaled in
    let aux (pp:natlt bh) (ii:natlt s) (j:natlt s)
      : Lemma (acc3 probs pp ii j
                 %~ acc3 (softmax_pages rm) pp ii j)
      = (* acc3 probs pp ii j == acc2 (flat3to2 probs) (pp*s+ii) j == acc2 sa' (pp*s+ii) j *)
        content_ok #f32 #bh #s #s p () probs pp ii j;
        (* row_corr: ra's row (pp*s+ii) == rm-page(pp)'s row ii (as lseq) *)
        row_corr #real #bh #s p () rscaled pp ii;
        (* lift to chest1 rows, so softmax_real of the equal rows agree, which is
           exactly acc2 (row_softmax_real ra) (pp*s+ii) j == acc3 (softmax_pages rm) pp ii j *)
        chest2_row_cong ra (slice_page rm pp) (pp*s+ii) ii
    in
    Classical.forall_intro_3 aux;
    lemma_approximates_intro3 probs (softmax_pages rm)
#pop-options

(* ----------------------------------------------------------------------- *)
(* The orchestrator.                                                        *)
(* ----------------------------------------------------------------------- *)

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn sdpa
  (bh s d : szp)
  (gQ  : array3 f32 (l3_batched_row_major bh s d) { is_global gQ })
  (gKT : array3 f32 (l3_batched_row_major bh d s) { is_global gKT })
  (gV  : array3 f32 (l3_batched_row_major bh s d) { is_global gV })
  (gScores : array3 f32 (l3_batched_row_major bh s s) { is_global gScores })
  (gOut : array3 f32 (l3_batched_row_major bh s d) { is_global gOut })
  (#sQ  : chest3 f32 bh s d)
  (#sKT : chest3 f32 bh d s)
  (#sV  : chest3 f32 bh s d)
  (#sScores0 : chest3 f32 bh s s)
  (#sOut0 : chest3 f32 bh s d)
  (rQ : erased (chest3 real bh s d))
  (rKT : erased (chest3 real bh d s))
  (rV : erased (chest3 real bh s d))
  (#fQ #fKT #fV : perm)
  preserves
    cpu **
    on gpu_loc (gQ |-> Frac fQ sQ ** gKT |-> Frac fKT sKT ** gV |-> Frac fV sV) **
    pure (sQ %~ rQ /\ sKT %~ rKT /\ sV %~ rV)
  requires
    on gpu_loc (gScores |-> sScores0) **
    on gpu_loc (gOut |-> sOut0) **
    pure (
      s * s <= max_blocks * max_threads /\
      SZ.fits (bh * (s * d)) /\
      SZ.fits (bh * (d * s)) /\
      SZ.fits (bh * (s * s)) /\
      bh * s * s <= max_blocks * max_threads /\
      bh * s <= max_blocks /\
      (bh * s) * s <= max_blocks * max_threads /\
      bh * (s * d) <= max_blocks * max_threads
    )
  ensures
    (exists* (probs : chest3 f32 bh s s) (eOut : chest3 f32 bh s d).
      on gpu_loc (gScores |-> probs) **
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ real_sdpa rQ rKT rV))
{
  let scale = sdpa_scale_f32 d;
  sdpa_scale_approx d;
  let rscores : erased (chest3 real bh s s) =
    hide (batched_matmul rQ rKT);
  let rscaled : erased (chest3 real bh s s) =
    hide (mscale (real_sdpa_scale d) (reveal rscores));
  (* ---- Step 1: gScores := Q @ K^T  (exact float matmul) ---- *)
  batched_gemm_f32 bh s d s gQ gKT gScores;
  batched_matmul_approx (reveal sQ) (reveal sKT) rQ rKT;
  (* on gpu_loc (gScores |-> batched_matmul sQ sKT) *)

  (* Machine products (fit by the preconditions: bh*s <= max_blocks and
     bh*s*s <= max_blocks*max_threads). *)
  let bhs : szp = bh *^ s;
  assert pure (SZ.v bhs == SZ.v bh * SZ.v s);
  let bhss : szp = bhs *^ s;
  assert pure (SZ.v bhss == SZ.v bh * SZ.v s * SZ.v s);
  let p1 : erased nat = SZ.v bhss;
  let p2 : erased nat = SZ.v bhs;

  (* ---- Step 2: gScores *= scale  (exact, in place over bh*s*s cells) ---- *)
  map_loc gpu_loc (fun () -> reshape3to1 p1 gScores);
  (* gScores re-interpreted as flat Array1 view over the same buffer. *)

  SMul.smul_fw_f32 scale bhss (from_array (l1_forward p1) (core gScores));
  (* view |-> lseq_map (mul scale) (from1 (to3 (batched_matmul sQ sKT))) *)

  smul_reshape_eq scale #bh #s #s p1 () (batched_matmul sQ sKT);
  mscale_approx scale (real_sdpa_scale d)
    (batched_matmul (reveal sQ) (reveal sKT)) (reveal rscores);
  map_loc gpu_loc (fun () ->
    reshape1to3_eq p1 gScores
      #(mscale scale (batched_matmul sQ sKT))
      #_
      #(chest_map (mul scale)
          (from_seq (l1_forward p1)
             (to_seq (l3_batched_row_major bh s s) (batched_matmul sQ sKT)))));
  (* on gpu_loc (gScores |-> mscale scale (batched_matmul sQ sKT)) *)

  (* ---- Step 3: row-softmax along the last (S) dim ---- *)
  map_loc gpu_loc (fun () -> reshape3to2 p2 gScores);
  (* gScores re-interpreted as (bh*s, s) Array2 view over the same buffer. *)

  reshape3to2_approx #bh #s #s p2 ()
    (mscale scale (batched_matmul (reveal sQ) (reveal sKT)))
    (reveal rscaled);

  RSI.row_softmax_rm_f32 bhs s max_threads
    (from_array (l2_row_major p2 s) (core gScores))
    (from_seq (l2_row_major p2 s)
       (to_seq (l3_batched_row_major bh s s) (reveal rscaled)));
  with sa'. assert on gpu_loc
    (from_array (l2_row_major p2 s) (core gScores) |-> sa');
  (* sa' %~ row_softmax_real (to_real (flat scaled scores)) *)

  (* Re-interpret the (bh*s, s) softmax result back as the (bh,s,s) tensor. *)
  let probs : chest3 f32 bh s s =
    from_seq (l3_batched_row_major bh s s)
      (to_seq (l2_row_major p2 s) sa');
  (* from_to2: from2 (to2 sa') == sa', and to3 probs == to2 sa' (A3.to_from),
     so flat3to2 probs == sa'. *)
  from_to2 (l2_row_major p2 s) sa';
  softmax_corr #bh #s p2 ()
    (reveal rscaled) sa' probs () ();
  map_loc gpu_loc (fun () -> reshape2to3_eq p2 gScores #probs #_ #sa');
  (* on gpu_loc (gScores |-> probs)
     ** probs %~ softmax_pages (to_real_matrix (mscale scale (batched_matmul sQ sKT))) *)

  (* ---- Step 4: gOut := probs @ V  (exact float matmul) ---- *)
  batched_gemm_f32 bh s s d gScores gV gOut;
  batched_matmul_approx probs (reveal sV)
    (softmax_pages (reveal rscaled)) rV;
  (* on gpu_loc (gOut |-> batched_matmul probs sV) *)

  ()
}
#pop-options

let sdpa_f32 = sdpa
