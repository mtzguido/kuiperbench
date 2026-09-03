module Kuiper.KB.SDPA

(* Implementation: see Kuiper.KB.SDPA.fsti for the design overview.

   KernelBench L1 #97 — Scaled Dot-Product Attention.

     out = softmax((Q @ K^T) / sqrt(D)) @ V

   The public entry retains the original rank-4 (B,H,S,D) tensors and proves
   their row-major collapse to the rank-3 (B*H,S,D) tensors consumed by the
   orchestrator.  The orchestrator then chains four already-verified Kuiper
   primitives over the SAME GPU buffers (no fused kernel, no data movement
   beyond what each primitive performs):

     1. reinterpret row-major K as a page-wise transposed column-major tensor,
        then batched GEMM (gScores := Q @ K^T, exact)
     2. ghost-reshape (bh,s,s) Array3 -> (bh*s*s) Array1, scalar-multiply by
        [scale] in place via smul_fw_f32, ghost-reshape back     (exact)
     3. ghost-reshape (bh,s,s) Array3 -> (bh*s, s) Array2, row-softmax along
        the last dim via Klas.RowSoftmax.row_softmax_rm_f32, reshape back
                                                                (the %~ step)
     4. batched_gemm_f32 bh s s d gScores gV gOut  (gOut := probs @ V, exact)

   The ghost reshapes are pure re-interpretations of the SAME buffer; they are
   the analogues of the Array3<->Array2 reshapes in Kuiper.KB.MatmulND, plus a
   new Array3<->Array1 reshape for the scalar-multiply.

   The direct real scale proof uses the temporary
   [Kuiper.KB.Compat.RsqrtApprox.rsqrt_approx] compatibility assumption. *)

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
module BG = Kuiper.Kernel.GEMM.Naive2
module RsqrtApprox = Kuiper.KB.Compat.RsqrtApprox
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
(* Logical page transpose.  A row-major (bh,s,d) tensor and a              *)
(* batched-column-major (bh,d,s) tensor have identical serialization when  *)
(* their last two logical axes are transposed.  The ghost conversions below *)
(* change only the verified view of K; extraction emits no transpose or     *)
(* memory operation.                                                        *)
(* ----------------------------------------------------------------------- *)

let transpose_pages
  (#et : Type) (#bh #rows #cols : nat)
  (m : chest3 et bh rows cols)
  : chest3 et bh cols rows =
  mk3 (fun p j i -> acc3 m p i j)

#push-options "--fuel 2 --ifuel 2 --z3rlimit 60"
let imap_swap_pages_rc
  (#bh #rows #cols : nat)
  (prc : abs (bh @| rows @| cols @| INil))
  : Lemma
      ((l3_batched_row_major bh rows cols).imap.f prc ==
       (l3_batched_col_major bh cols rows).imap.f
         (prc._1, (prc._2._2._1, (prc._2._1, ()))))
  = ()

let imap_swap_pages_cr
  (#bh #rows #cols : nat)
  (prc : abs (bh @| rows @| cols @| INil))
  : Lemma
      ((l3_batched_col_major bh rows cols).imap.f prc ==
       (l3_batched_row_major bh cols rows).imap.f
         (prc._1, (prc._2._2._1, (prc._2._1, ()))))
  = ()
#pop-options

#push-options "--fuel 2 --ifuel 2 --z3rlimit 50"
let transpose_pages_index
  (#et : Type) (#bh #rows #cols : nat)
  (m : chest3 et bh rows cols)
  (i : natlt (sizeof (bh @| rows @| cols @| INil)))
  : Lemma
      (Seq.index (to_seq (l3_batched_row_major bh rows cols) m) i ==
       Seq.index
         (to_seq (l3_batched_col_major bh cols rows) (transpose_pages m)) i)
  = let lr = l3_batched_row_major bh rows cols in
    let lc = l3_batched_col_major bh cols rows in
    let prc : abs (bh @| rows @| cols @| INil) = inverse_f lr.imap i in
    let pcr : abs (bh @| cols @| rows @| INil) =
      (prc._1, (prc._2._2._1, (prc._2._1, ()))) in
    inverse_lem lr.imap i;
    imap_swap_pages_rc #bh #rows #cols prc;
    assert (lc.imap.f pcr == i);
    lem_pat lc.imap pcr;
    assert (inverse_f lc.imap i == pcr);
    assert (acc (transpose_pages m) pcr == acc m prc)

let transpose_pages_to_seq
  (#et : Type) (#bh #rows #cols : nat)
  (m : chest3 et bh rows cols)
  : Lemma
      (Seq.equal
        (to_seq (l3_batched_row_major bh rows cols) m)
        (to_seq
          (l3_batched_col_major bh cols rows) (transpose_pages m)))
  = introduce forall (i : natlt (sizeof (bh @| rows @| cols @| INil))).
      Seq.index (to_seq (l3_batched_row_major bh rows cols) m) i ==
      Seq.index
        (to_seq (l3_batched_col_major bh cols rows) (transpose_pages m)) i
    with transpose_pages_index m i

let transpose_pages_index_back
  (#et : Type) (#bh #rows #cols : nat)
  (m : chest3 et bh rows cols)
  (i : natlt (sizeof (bh @| rows @| cols @| INil)))
  : Lemma
      (Seq.index (to_seq (l3_batched_col_major bh rows cols) m) i ==
       Seq.index
         (to_seq (l3_batched_row_major bh cols rows) (transpose_pages m)) i)
  = let lc = l3_batched_col_major bh rows cols in
    let lr = l3_batched_row_major bh cols rows in
    let prc : abs (bh @| rows @| cols @| INil) = inverse_f lc.imap i in
    let pcr : abs (bh @| cols @| rows @| INil) =
      (prc._1, (prc._2._2._1, (prc._2._1, ()))) in
    inverse_lem lc.imap i;
    imap_swap_pages_cr #bh #rows #cols prc;
    assert (lr.imap.f pcr == i);
    lem_pat lr.imap pcr;
    assert (inverse_f lr.imap i == pcr);
    assert (acc (transpose_pages m) pcr == acc m prc)

let transpose_pages_to_seq_back
  (#et : Type) (#bh #rows #cols : nat)
  (m : chest3 et bh rows cols)
  : Lemma
      (Seq.equal
        (to_seq (l3_batched_col_major bh rows cols) m)
        (to_seq
          (l3_batched_row_major bh cols rows) (transpose_pages m)))
  = introduce forall (i : natlt (sizeof (bh @| rows @| cols @| INil))).
      Seq.index (to_seq (l3_batched_col_major bh rows cols) m) i ==
      Seq.index
        (to_seq (l3_batched_row_major bh cols rows) (transpose_pages m)) i
    with transpose_pages_index_back m i
#pop-options

ghost
fn transpose_pages_view
  (#et : Type) (#bh #rows #cols : nat)
  (g : array3 et (l3_batched_row_major bh rows cols))
  (#m : chest3 et bh rows cols) (#f : perm)
  requires g |-> Frac f m
  ensures
    from_array (l3_batched_col_major bh cols rows) (core g)
      |-> Frac f (transpose_pages m)
{
  tensor_concr g;
  transpose_pages_to_seq m;
  rewrite
    (core g |-> Frac f (to_seq (l3_batched_row_major bh rows cols) m))
  as
    (core g |-> Frac f
      (to_seq (l3_batched_col_major bh cols rows) (transpose_pages m)));
  tensor_abs (l3_batched_col_major bh cols rows) (core g)
    #f #(transpose_pages m)
}

ghost
fn transpose_pages_view_back
  (#et : Type) (#bh #rows #cols : nat)
  (g : array3 et (l3_batched_row_major bh rows cols))
  (#m : chest3 et bh cols rows) (#f : perm)
  requires
    from_array (l3_batched_col_major bh cols rows) (core g) |-> Frac f m
  ensures g |-> Frac f (transpose_pages m)
{
  tensor_concr
    (from_array (l3_batched_col_major bh cols rows) (core g));
  transpose_pages_to_seq_back m;
  rewrite
    (core (from_array (l3_batched_col_major bh cols rows) (core g))
      |-> Frac f (to_seq (l3_batched_col_major bh cols rows) m))
  as
    (core g |-> Frac f
      (to_seq (l3_batched_col_major bh cols rows) m));
  rewrite
    (core g |-> Frac f
      (to_seq (l3_batched_col_major bh cols rows) m))
  as
    (core g |-> Frac f
      (to_seq (l3_batched_row_major bh rows cols) (transpose_pages m)));
  tensor_abs (l3_batched_row_major bh rows cols) (core g)
    #f #(transpose_pages m);
  rewrite
    (from_array (l3_batched_row_major bh rows cols) (core g)
      |-> Frac f (transpose_pages m))
  as
    (g |-> Frac f (transpose_pages m))
}

let transpose_pages_involutive
  (#et : Type) (#bh #rows #cols : nat)
  (m : chest3 et bh rows cols)
  : Lemma (transpose_pages (transpose_pages m) == m)
  = introduce forall (idx : abs (bh @| rows @| cols @| INil)).
      acc (transpose_pages (transpose_pages m)) idx == acc m idx
    with (let (p, (i, (j, ()))) = idx in ());
    Kuiper.Chest.lemma_equal_intro (transpose_pages (transpose_pages m)) m;
    Kuiper.Chest.ext (transpose_pages (transpose_pages m)) m

let transpose_pages_approx
  (#et : Type) {| scalar et, real_like et |}
  (#bh #rows #cols : nat)
  (m : chest3 et bh rows cols)
  (rm : chest3 real bh rows cols)
  : Lemma
      (requires m %~ rm)
      (ensures transpose_pages m %~ transpose_pages rm)
  = introduce forall (idx : abs (bh @| cols @| rows @| INil)).
      acc (transpose_pages m) idx %~ acc (transpose_pages rm) idx
    with (let (p, (j, (i, ()))) = idx in
          assert (acc3 m p i j %~ acc3 rm p i j))

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

(* The public rank-4 ABI and the internal rank-3 batched ABI have identical
   row-major offsets after encoding page [(bi,hi)] as [bi*h+hi]. *)
#push-options "--z3rlimit 40"
let imap_eq4to3
  (#b #h #s #d : nat)
  (bi:nat{bi<b}) (hi:nat{hi<h}) (si:nat{si<s}) (di:nat{di<d})
  : Lemma
    ((l4_batched_row_major b h s d).imap.f (bi,(hi,(si,(di,()))))
       == (l3_batched_row_major (b*h) s d).imap.f
            ((bi*h+hi),(si,(di,()))))
  = let l_d : layout_f_for (d @| INil) = major_on 0 d lunit in
    let l_sd : layout_f_for (s @| d @| INil) = major_on 0 s l_d in
    let l_hsd : layout_f_for (h @| s @| d @| INil) = major_on 0 h l_sd in
    FStar.Math.Lemmas.lemma_mult_le_right h (bi+1) b;
    FStar.Math.Lemmas.distributivity_add_left bi 1 h;
    let page : natlt (b*h) = bi*h + hi in
    assert ((l4_batched_row_major b h s d).imap.f
              (bi,(hi,(si,(di,()))))
              == (major_on 0 b l_hsd).f (bi,(hi,(si,(di,())))));
    assert ((l3_batched_row_major (b*h) s d).imap.f
              (page,(si,(di,())))
              == (major_on 0 (b*h) l_sd).f (page,(si,(di,()))));
    major_on_zero_f #3 b #(h @| s @| d @| INil) l_hsd
      bi (hi,(si,(di,())));
    major_on_zero_f #2 h #(s @| d @| INil) l_sd hi (si,(di,()));
    major_on_zero_f #2 (b*h) #(s @| d @| INil) l_sd
      page (si,(di,()));
    assert (sizeof (h @| s @| d @| INil) ==
              h * sizeof (s @| d @| INil));
    FStar.Math.Lemmas.paren_mul_right bi h (sizeof (s @| d @| INil));
    FStar.Math.Lemmas.distributivity_add_left
      (bi*h) hi (sizeof (s @| d @| INil))
#pop-options

let flatten_bh_index
  (#et:Type) (#b #h #s #d : nat)
  (x : chest4 et b h s d)
  (bi:nat{bi<b}) (hi:nat{hi<h}) (si:nat{si<s}) (di:nat{di<d})
  (page : natlt (b*h)) (_:squash (page == bi*h+hi))
  : Lemma (acc3 (flatten_bh x) page si di == acc4 x bi hi si di)
  = imap_eq4to3 #b #h #s #d bi hi si di;
    let l4 = l4_batched_row_major b h s d in
    let l3 = l3_batched_row_major (b*h) s d in
    let bytes = to_seq l4 x in
    assert (acc3 (from_seq l3 bytes) page si di ==
              Seq.index bytes (l3.imap.f (page,(si,(di,())))));
    assert (l3.imap.f (page,(si,(di,()))) ==
              l4.imap.f (bi,(hi,(si,(di,())))));
    assert (inverse_f l4.imap (l4.imap.f (bi,(hi,(si,(di,()))))) ==
              (bi,(hi,(si,(di,())))))

let flatten_bh_index_p
  (#et:Type) (#b #h #s #d : nat)
  (p:nat) (_:squash (p == b*h))
  (x : chest4 et b h s d)
  (bi:nat{bi<b}) (hi:nat{hi<h}) (si:nat{si<s}) (di:nat{di<d})
  (page : natlt p) (_:squash (page == bi*h+hi))
  : Lemma
      (acc3
        (from_seq (l3_batched_row_major p s d)
          (to_seq (l4_batched_row_major b h s d) x))
        page si di == acc4 x bi hi si di)
  = imap_eq4to3 #b #h #s #d bi hi si di;
    let l4 = l4_batched_row_major b h s d in
    let l3 = l3_batched_row_major p s d in
    let bytes = to_seq l4 x in
    assert (l3.imap.f (page,(si,(di,()))) ==
              l4.imap.f (bi,(hi,(si,(di,())))));
    assert (acc3 (from_seq l3 bytes) page si di ==
              Seq.index bytes (l3.imap.f (page,(si,(di,())))));
    assert (inverse_f l4.imap (l4.imap.f (bi,(hi,(si,(di,()))))) ==
              (bi,(hi,(si,(di,())))))

let encode_bh_bound (b h : nat) (bi:nat{bi<b}) (hi:nat{hi<h})
  : Lemma (bi*h+hi < b*h)
  = FStar.Math.Lemmas.distributivity_add_left bi 1 h;
    FStar.Math.Lemmas.lemma_mult_le_right h (bi+1) b

let div_bound (a:nat) (n:pos) (m:nat)
  : Lemma (requires a < m * n) (ensures a / n < m)
  = FStar.Math.Lemmas.multiply_fractions a n;
    FStar.Math.Lemmas.swap_mul n (a/n);
    FStar.Math.Lemmas.multiplication_order_lemma (a/n) m n

let decode_bh (b:nat) (h:pos) (page:nat{page < b*h})
  : Lemma (ensures (
      let hi = page % h in let bi = page / h in
      bi < b /\ hi < h /\ page == bi*h+hi))
  = let hi = page % h in
    let bi = page / h in
    FStar.Math.Lemmas.lemma_mod_lt page h;
    div_bound page h b;
    FStar.Math.Lemmas.lemma_div_mod page h;
    FStar.Math.Lemmas.swap_mul h bi

let from_to3
  (#et:Type) (#d0 #d1 #d2:nat)
  (l : full_layout3 d0 d1 d2)
  (x : chest3 et d0 d1 d2)
  : Lemma (from_seq l (to_seq l x) == x)
  = let lhs = from_seq l (to_seq l x) in
    let aux (i:natlt d0) (j:natlt d1) (k:natlt d2)
      : Lemma (acc3 lhs i j k == acc3 x i j k)
      = inverse_lem l.imap (l.imap.f (i,(j,(k,()))));
        l.imap.is_inj
          (inverse_f l.imap (l.imap.f (i,(j,(k,()))))) (i,(j,(k,())))
    in
    Classical.forall_intro_3 aux;
    Kuiper.Chest.lemma_equal_intro lhs x;
    Kuiper.Chest.ext lhs x

(* Runtime-free Array4 <-> Array3 views used only inside the public entry. *)
ghost
fn reshape4to3
  (#et:Type) (#b #h #s #d : nat)
  (p : nat) (#_ : squash (p == b * h))
  (a4 : array4 et (l4_batched_row_major b h s d))
  (#x4 : chest4 et b h s d) (#f : perm)
  requires a4 |-> Frac f x4
  ensures
    from_array (l3_batched_row_major p s d) (core a4)
      |-> Frac f (from_seq (l3_batched_row_major p s d)
                     (to_seq (l4_batched_row_major b h s d) x4))
{
  tensor_concr a4;
  tensor_abs' (l3_batched_row_major p s d) (core a4)
}

ghost
fn reshape3to4
  (#et:Type) (#b #h #s #d : nat)
  (p : nat) (#_ : squash (p == b * h))
  (a4 : array4 et (l4_batched_row_major b h s d))
  (#x4 : chest4 et b h s d) (#f : perm)
  requires
    from_array (l3_batched_row_major p s d) (core a4)
      |-> Frac f (from_seq (l3_batched_row_major p s d)
                     (to_seq (l4_batched_row_major b h s d) x4))
  ensures a4 |-> Frac f x4
{
  tensor_concr
    (from_array (l3_batched_row_major p s d) (core a4));
  rewrite
    (core (from_array (l3_batched_row_major p s d) (core a4))
      |-> Frac f (to_seq (l3_batched_row_major p s d)
                    (from_seq (l3_batched_row_major p s d)
                      (to_seq (l4_batched_row_major b h s d) x4))))
  as
    (core a4 |-> Frac f
      (to_seq (l4_batched_row_major b h s d) x4));
  tensor_abs (l4_batched_row_major b h s d) (core a4) #f #x4;
  rewrite
    (from_array (l4_batched_row_major b h s d) (core a4)
      |-> Frac f x4)
  as (a4 |-> Frac f x4)
}

ghost
fn reshape3to4_eq
  (#et:Type) (#b #h #s #d : nat)
  (p : nat) (#_ : squash (p == b * h))
  (a4 : array4 et (l4_batched_row_major b h s d))
  (#x4 : chest4 et b h s d) (#f : perm)
  (#x3 : chest3 et p s d)
  (#_ : squash (x3 ==
    from_seq (l3_batched_row_major p s d)
      (to_seq (l4_batched_row_major b h s d) x4)))
  requires
    from_array (l3_batched_row_major p s d) (core a4) |-> Frac f x3
  ensures a4 |-> Frac f x4
{
  rewrite
    (from_array (l3_batched_row_major p s d) (core a4) |-> Frac f x3)
  as
    (from_array (l3_batched_row_major p s d) (core a4)
      |-> Frac f (from_seq (l3_batched_row_major p s d)
                     (to_seq (l4_batched_row_major b h s d) x4)));
  reshape3to4 p a4 #x4 #f
}

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

let lemma_approximates_intro4
  (#et:Type) {| scalar et, real_like et |} (#d0 #d1 #d2 #d3 : nat)
  (m1 : chest4 et d0 d1 d2 d3) (m2 : chest4 real d0 d1 d2 d3)
  : Lemma
      (requires forall (i:natlt d0) (j:natlt d1)
                       (k:natlt d2) (l:natlt d3).
                  acc4 m1 i j k l %~ acc4 m2 i j k l)
      (ensures m1 %~ m2)
  = introduce forall
      (idx : abs (d0 @| d1 @| d2 @| d3 @| INil)). acc m1 idx %~ acc m2 idx
    with (let (i, (j, (k, (l, ())))) = idx in
          assert (acc4 m1 i j k l %~ acc4 m2 i j k l))

(* Move approximation facts across the verified rank-4/rank-3 page view. *)
let flatten_bh_approx
  (#et:Type) {| scalar et, real_like et |} (#b #h #s #d : pos)
  (m1 : chest4 et b h s d) (m2 : chest4 real b h s d)
  : Lemma (requires m1 %~ m2)
          (ensures flatten_bh m1 %~ flatten_bh m2)
  = let aux (page:natlt (b*h)) (si:natlt s) (di:natlt d)
      : Lemma
          (acc3 (flatten_bh m1) page si di %~
           acc3 (flatten_bh m2) page si di)
      = decode_bh b h page;
        let hi = page % h in
        let bi = page / h in
        flatten_bh_index m1 bi hi si di page ();
        flatten_bh_index m2 bi hi si di page ()
    in
    Classical.forall_intro_3 aux;
    lemma_approximates_intro3 (flatten_bh m1) (flatten_bh m2)

let unflatten_bh_approx
  (#et:Type) {| scalar et, real_like et |} (#b #h #s #d : pos)
  (m1 : chest3 et (b*h) s d) (m2 : chest3 real (b*h) s d)
  : Lemma (requires m1 %~ m2)
          (ensures unflatten_bh m1 %~ unflatten_bh m2)
  = let aux (bi:natlt b) (hi:natlt h) (si:natlt s) (di:natlt d)
      : Lemma
          (acc4 (unflatten_bh m1) bi hi si di %~
           acc4 (unflatten_bh m2) bi hi si di)
      = encode_bh_bound b h bi hi;
        let page : natlt (b*h) = bi*h+hi in
        let l4 = l4_batched_row_major b h s d in
        let l3 = l3_batched_row_major (b*h) s d in
        imap_eq4to3 #b #h #s #d bi hi si di;
        assert (acc4 (from_seq l4 (to_seq l3 m1)) bi hi si di ==
                  acc3 m1 page si di);
        assert (acc4 (from_seq l4 (to_seq l3 m2)) bi hi si di ==
                  acc3 m2 page si di)
    in
    Classical.forall_intro_4 aux;
    lemma_approximates_intro4 (unflatten_bh m1) (unflatten_bh m2)

let flatten_bh_approx_p
  (#et:Type) {| scalar et, real_like et |} (#b #h #s #d : pos)
  (p:nat) (_:squash (p == b*h))
  (m1 : chest4 et b h s d) (m2 : chest4 real b h s d)
  : Lemma (requires m1 %~ m2)
      (ensures
        from_seq (l3_batched_row_major p s d)
          (to_seq (l4_batched_row_major b h s d) m1)
        %~
        from_seq (l3_batched_row_major p s d)
          (to_seq (l4_batched_row_major b h s d) m2))
  = let x1 = from_seq (l3_batched_row_major p s d)
      (to_seq (l4_batched_row_major b h s d) m1) in
    let x2 = from_seq (l3_batched_row_major p s d)
      (to_seq (l4_batched_row_major b h s d) m2) in
    let aux (page:natlt p) (si:natlt s) (di:natlt d)
      : Lemma (acc3 x1 page si di %~ acc3 x2 page si di)
      = assert (page < b*h);
        decode_bh b h page;
        let hi = page % h in
        let bi = page / h in
        flatten_bh_index_p p () m1 bi hi si di page ();
        flatten_bh_index_p p () m2 bi hi si di page ()
    in
    Classical.forall_intro_3 aux;
    lemma_approximates_intro3 x1 x2

let unflatten_bh_approx_p
  (#et:Type) {| scalar et, real_like et |} (b h : pos) (#s #d : pos)
  (p:nat) (_:squash (p == b*h))
  (m1 : chest3 et p s d) (m2 : chest3 real p s d)
  : Lemma (requires m1 %~ m2)
      (ensures
        from_seq (l4_batched_row_major b h s d)
          (to_seq (l3_batched_row_major p s d) m1)
        %~
        from_seq (l4_batched_row_major b h s d)
          (to_seq (l3_batched_row_major p s d) m2))
  = let x1 = from_seq (l4_batched_row_major b h s d)
      (to_seq (l3_batched_row_major p s d) m1) in
    let x2 = from_seq (l4_batched_row_major b h s d)
      (to_seq (l3_batched_row_major p s d) m2) in
    let aux (bi:natlt b) (hi:natlt h) (si:natlt s) (di:natlt d)
      : Lemma (acc4 x1 bi hi si di %~ acc4 x2 bi hi si di)
      = encode_bh_bound b h bi hi;
        let page : natlt p = bi*h+hi in
        let l4 = l4_batched_row_major b h s d in
        let l3 = l3_batched_row_major p s d in
        imap_eq4to3 #b #h #s #d bi hi si di;
        assert (l4.imap.f (bi,(hi,(si,(di,())))) ==
                  l3.imap.f (page,(si,(di,()))));
        assert (acc4 x1 bi hi si di == acc3 m1 page si di);
        assert (acc4 x2 bi hi si di == acc3 m2 page si di)
    in
    Classical.forall_intro_4 aux;
    lemma_approximates_intro4 x1 x2

let sdpa_scale_approx (d : szp)
  : Lemma (sdpa_scale_f32 d %~ real_sdpa_scale d)
  = let d64 : Int64.t = FStar.Int.Cast.uint64_to_int64
      (FStar.SizeT.sizet_to_uint64 d) in
    assert (Int64.v d64 == SZ.v d);
    let df : f32 = of_int d64 in
    of_int_approx #f32 d64;
    assert (df %~ FStar.Real.of_int d);
    RsqrtApprox.rsqrt_approx df (FStar.Real.of_int d)

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

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn sdpa
  (bh s d : szp)
  (gQ  : array3 f32 (l3_batched_row_major bh s d) { is_global gQ })
  (gK  : array3 f32 (l3_batched_row_major bh s d) { is_global gK })
  (gV  : array3 f32 (l3_batched_row_major bh s d) { is_global gV })
  (gScores : array3 f32 (l3_batched_row_major bh s s) { is_global gScores })
  (gOut : array3 f32 (l3_batched_row_major bh s d) { is_global gOut })
  (#sQ  : chest3 f32 bh s d)
  (#sK  : chest3 f32 bh s d)
  (#sV  : chest3 f32 bh s d)
  (#sScores0 : chest3 f32 bh s s)
  (#sOut0 : chest3 f32 bh s d)
  (rQ : erased (chest3 real bh s d))
  (rK : erased (chest3 real bh s d))
  (rV : erased (chest3 real bh s d))
  (#fQ #fK #fV : perm)
  preserves
    cpu **
    on gpu_loc (gQ |-> Frac fQ sQ ** gK |-> Frac fK sK ** gV |-> Frac fV sV) **
    pure (sQ %~ rQ /\ sK %~ rK /\ sV %~ rV)
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
      pure (eOut %~ real_sdpa rQ rK rV))
{
  let scale = sdpa_scale_f32 d;
  sdpa_scale_approx d;
  transpose_pages_approx (reveal sK) rK;
  let rscores : erased (chest3 real bh s s) =
    hide (batched_matmul rQ (transpose_pages rK));
  let rscaled : erased (chest3 real bh s s) =
    hide (mscale (real_sdpa_scale d) (reveal rscores));
  (* ---- Step 1: gScores := Q @ K^T  (exact float matmul) ----
     The row-major K buffer is logically transposed by viewing it as a
     batched-column-major (bh,d,s) tensor; this is a proof-only operation. *)
  map_loc gpu_loc (fun () -> transpose_pages_view gK);
  BG.bmmcomb_gpu_exact #f32 MS.comb2 bh s s d
    #(l3_batched_row_major bh s d)
    #(l3_batched_col_major bh d s)
    #(l3_batched_row_major bh s s)
    gQ
    (from_array (l3_batched_col_major bh d s) (core gK))
    gScores;
  with scores'. assert on gpu_loc (gScores |-> scores');
  assert pure
    (equal scores' (batched_matmul sQ (transpose_pages sK)));
  map_loc gpu_loc (fun () ->
    transpose_pages_view_back gK #(transpose_pages sK));
  transpose_pages_involutive sK;
  batched_matmul_approx
    (reveal sQ) (transpose_pages (reveal sK))
    rQ (transpose_pages rK);
  (* on gpu_loc
       (gScores |-> batched_matmul sQ (transpose_pages sK)) *)

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
  (* view |-> map (mul scale) (flatten (Q @ transpose_pages K)) *)

  smul_reshape_eq scale #bh #s #s p1 ()
    (batched_matmul sQ (transpose_pages sK));
  mscale_approx scale (real_sdpa_scale d)
    (batched_matmul (reveal sQ) (transpose_pages (reveal sK)))
    (reveal rscores);
  map_loc gpu_loc (fun () ->
    reshape1to3_eq p1 gScores
      #(mscale scale (batched_matmul sQ (transpose_pages sK)))
      #_
      #(chest_map (mul scale)
          (from_seq (l1_forward p1)
             (to_seq (l3_batched_row_major bh s s)
               (batched_matmul sQ (transpose_pages sK))))));
  (* on gpu_loc (gScores |-> mscale scale (Q @ transpose_pages K)) *)

  (* ---- Step 3: row-softmax along the last (S) dim ---- *)
  map_loc gpu_loc (fun () -> reshape3to2 p2 gScores);
  (* gScores re-interpreted as (bh*s, s) Array2 view over the same buffer. *)

  reshape3to2_approx #bh #s #s p2 ()
    (mscale scale
      (batched_matmul (reveal sQ) (transpose_pages (reveal sK))))
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
     ** probs %~ softmax_pages (mscale scale (Q @ transpose_pages K)) *)

  (* ---- Step 4: gOut := probs @ V  (exact float matmul) ---- *)
  batched_gemm_f32 bh s s d gScores gV gOut;
  batched_matmul_approx probs (reveal sV)
    (softmax_pages (reveal rscaled)) rV;
  (* on gpu_loc (gOut |-> batched_matmul probs sV) *)

  ()
}
#pop-options

inline_for_extraction noextract
fn sdpa_alloc
  (bh s d : szp)
  (gQ : array3 f32 (l3_batched_row_major bh s d) { is_global gQ })
  (gK : array3 f32 (l3_batched_row_major bh s d) { is_global gK })
  (gV : array3 f32 (l3_batched_row_major bh s d) { is_global gV })
  (#sQ #sK #sV : chest3 f32 bh s d)
  (rQ rK rV : erased (chest3 real bh s d))
  (#fQ #fK #fV : perm)
  preserves
    cpu **
    on gpu_loc
      (gQ |-> Frac fQ sQ **
       gK |-> Frac fK sK **
       gV |-> Frac fV sV) **
    pure (sQ %~ rQ /\ sK %~ rK /\ sV %~ rV)
  requires
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
  returns gOut : array3 f32 (l3_batched_row_major bh s d)
  ensures
    (exists* (eOut : chest3 f32 bh s d).
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ real_sdpa rQ rK rV))
{
  let bhs : szp = bh *^ s;
  let bhss : szp = bhs *^ s;
  let bhsd : szp = bhs *^ d;
  let gScores = alloc0 #f32 bhss
    (l3_batched_row_major bh s s);
  let gOut = alloc0 #f32 bhsd
    (l3_batched_row_major bh s d);
  sdpa bh s d gQ gK gV gScores gOut rQ rK rV;
  free gScores;
  gOut
}

fn sdpa_f32
  (b h s d : szp)
  (gQ : array4 f32 (l4_batched_row_major b h s d) { is_global gQ })
  (gK : array4 f32 (l4_batched_row_major b h s d) { is_global gK })
  (gV : array4 f32 (l4_batched_row_major b h s d) { is_global gV })
  (#sQ #sK #sV : chest4 f32 b h s d)
  (rQ rK rV : erased (chest4 real b h s d))
  (#fQ #fK #fV : perm)
  preserves
    cpu **
    on gpu_loc
      (gQ |-> Frac fQ sQ **
       gK |-> Frac fK sK **
       gV |-> Frac fV sV) **
    pure (sQ %~ rQ /\ sK %~ rK /\ sV %~ rV)
  requires
    pure (
      SZ.fits (b * h) /\
      s * s <= max_blocks * max_threads /\
      SZ.fits ((b * h) * (s * d)) /\
      SZ.fits ((b * h) * (d * s)) /\
      SZ.fits ((b * h) * (s * s)) /\
      (b * h) * s * s <= max_blocks * max_threads /\
      (b * h) * s <= max_blocks /\
      ((b * h) * s) * s <= max_blocks * max_threads /\
      (b * h) * (s * d) <= max_blocks * max_threads
    )
  returns gOut : array4 f32 (l4_batched_row_major b h s d)
  ensures
    (exists* (eOut : chest4 f32 b h s d).
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ real_sdpa4 rQ rK rV))
{
  let bh = b *^ h;
  assert pure (SZ.v bh == SZ.v b * SZ.v h);

  let rQ3 : chest3 real (SZ.v bh) s d =
    from_seq (l3_batched_row_major (SZ.v bh) s d)
      (to_seq (l4_batched_row_major b h s d) rQ);
  let rK3 : chest3 real (SZ.v bh) s d =
    from_seq (l3_batched_row_major (SZ.v bh) s d)
      (to_seq (l4_batched_row_major b h s d) rK);
  let rV3 : chest3 real (SZ.v bh) s d =
    from_seq (l3_batched_row_major (SZ.v bh) s d)
      (to_seq (l4_batched_row_major b h s d) rV);
  flatten_bh_approx_p (SZ.v bh) () (reveal sQ) rQ;
  flatten_bh_approx_p (SZ.v bh) () (reveal sK) rK;
  flatten_bh_approx_p (SZ.v bh) () (reveal sV) rV;

  map_loc gpu_loc (fun () -> reshape4to3 (SZ.v bh) gQ);
  map_loc gpu_loc (fun () -> reshape4to3 (SZ.v bh) gK);
  map_loc gpu_loc (fun () -> reshape4to3 (SZ.v bh) gV);

  let sQ3 : chest3 f32 (SZ.v bh) s d =
    from_seq (l3_batched_row_major (SZ.v bh) s d)
      (to_seq (l4_batched_row_major b h s d) (reveal sQ));
  let sK3 : chest3 f32 (SZ.v bh) s d =
    from_seq (l3_batched_row_major (SZ.v bh) s d)
      (to_seq (l4_batched_row_major b h s d) (reveal sK));
  let sV3 : chest3 f32 (SZ.v bh) s d =
    from_seq (l3_batched_row_major (SZ.v bh) s d)
      (to_seq (l4_batched_row_major b h s d) (reveal sV));
  assert pure (sQ3 %~ rQ3 /\ sK3 %~ rK3 /\ sV3 %~ rV3);

  let bhs : szp = bh *^ s;
  let bhss : szp = bhs *^ s;
  let bhsd : szp = bhs *^ d;
  let gScores = alloc0 #f32 bhss
    (l3_batched_row_major (SZ.v bh) s s);
  let gOut4 = alloc0 #f32 bhsd
    (l4_batched_row_major b h s d);
  map_loc gpu_loc (fun () -> reshape4to3 (SZ.v bh) gOut4);
  sdpa bh s d
    (from_array (l3_batched_row_major (SZ.v bh) s d) (core gQ))
    (from_array (l3_batched_row_major (SZ.v bh) s d) (core gK))
    (from_array (l3_batched_row_major (SZ.v bh) s d) (core gV))
    gScores
    (from_array (l3_batched_row_major (SZ.v bh) s d) (core gOut4))
    rQ3 rK3 rV3;
  with eOut3.
    assert on gpu_loc
      (from_array (l3_batched_row_major (SZ.v bh) s d) (core gOut4)
        |-> eOut3);
  free gScores;

  map_loc gpu_loc (fun () -> reshape3to4 (SZ.v bh) gQ);
  map_loc gpu_loc (fun () -> reshape3to4 (SZ.v bh) gK);
  map_loc gpu_loc (fun () -> reshape3to4 (SZ.v bh) gV);

  let rOut3 : chest3 real (SZ.v bh) s d =
    real_sdpa rQ3 rK3 rV3;
  unflatten_bh_approx_p b h (SZ.v bh) () eOut3 rOut3;
  let eOut4 : chest4 f32 b h s d =
    from_seq (l4_batched_row_major b h s d)
      (to_seq (l3_batched_row_major (SZ.v bh) s d) eOut3);
  assert pure (eOut4 %~ real_sdpa4 rQ rK rV);

  Kuiper.Tensor.Layout.to_from
    (l4_batched_row_major b h s d)
    (to_seq (l3_batched_row_major (SZ.v bh) s d) eOut3);
  from_to3 (l3_batched_row_major (SZ.v bh) s d) eOut3;
  map_loc gpu_loc
    (fun () -> reshape3to4_eq (SZ.v bh) gOut4 #eOut4 #_ #eOut3);
  gOut4
}
