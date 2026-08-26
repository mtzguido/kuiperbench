module Kuiper.KB.Matmul4D

(* Implementation: see Kuiper.KB.Matmul4D.fsti for the design overview.

   KernelBench L1 #11 — 4-D tensor (b,i,j,l) @ matrix (l,k) -> (b,i,j,k),
   i.e. einsum("bijl,lk->bijk").  The SAME matrix B multiplies every
   (b,i,j) slice.

   Design A (maximal reuse): we collapse only the LEADING TWO dims of the
   4-D tensor to reach a 3-D tensor, via a pure ghost re-interpretation of
   the SAME GPU buffer (no data movement: Array4.lower + Array3.raise'), and
   then CALL the already-verified challenge-10 kernel [Kuiper.KB.MatmulND]
   on the (b*i, j, l) @ (l, k) product.  The result is re-interpreted back as
   the (b,i,j,k) tensor.

   Zero assume · zero magic · zero admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.EMatrix
module EMatrix = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module K = Kuiper.Kernel.GEMM.Naive3
open Kuiper.KB.MatmulND { flat3to2, matmul_nd, content_ok }
open Kuiper.Injection
open Kuiper.Shape

(* ----------------------------------------------------------------------- *)
(* Row-major flatten helpers (implementation-only — the interface spec is   *)
(* stated entirely via [ematmul4], so these live here rather than in the    *)
(* .fsti).  [flat4to3] collapses the leading two dims (b,i)->(b*i); the      *)
(* full [flat4to2] linearizes all leading dims into a (b*i*j, l) matrix      *)
(* (= [flat3to2 (flat4to3 s)], since [(b*i)*j == b*i*j]).                    *)
(* ----------------------------------------------------------------------- *)

let flat4to3 (#et:Type) (#b #i #j #l:nat) (s : chest4 et b i j l)
  : chest3 et (b*i) j l
  = from_seq (l3_batched_row_major (b*i) j l)
       (to_seq (l4_row_major b i j l) s)

let flat4to2 (#et:Type) (#b #i #j #l:nat) (s : chest4 et b i j l)
  : EMatrix.chest2 et (b*i*j) l
  = flat3to2 (flat4to3 s)

(* ----------------------------------------------------------------------- *)
(* [from_seq (to_seq s) == s] for Array3 — the "round-trip the other way"   *)
(* (the library only provides [to_seq (from_seq x) == x]).  Mirror of the   *)
(* [from_to2] lemma in MatmulND; proof by extensionality + layout injec-    *)
(* tivity.                                                                  *)
(* ----------------------------------------------------------------------- *)

let from_to3
  (#et:Type) (#d0 #d1 #d2:nat)
  (l : full_layout3 d0 d1 d2)
  (s : chest3 et d0 d1 d2)
  : Lemma (ensures from_seq l (to_seq l s) == s)
  = let lhs = from_seq l (to_seq l s) in
    let aux (i:natlt d0) (j:natlt d1) (c:natlt d2)
      : Lemma (acc3 lhs i j c == acc3 s i j c)
      = inverse_lem l.imap (l.imap.f (i,(j,(c,()))));
        l.imap.is_inj (inverse_f l.imap (l.imap.f (i,(j,(c,()))))) (i,(j,(c,())))
    in
    Classical.forall_intro_3 aux;
    Kuiper.Chest.lemma_equal_intro lhs s;
    Kuiper.Chest.ext lhs s

(* ----------------------------------------------------------------------- *)
(* Spec glue.  The 4-D flatten [flat4to2] (= [flat3to2] of [flat4to3]) is   *)
(* exactly the row-major linearization of all leading dims.  These two pure *)
(* lemmas bridge the pre/post conditions across the [SZ.v (b*^i)] vs [b*i]  *)
(* dimension gap.                                                           *)
(* ----------------------------------------------------------------------- *)

(* Forward: the [flat3to2] of the (p,j,ll) 3-D view of [s] equals [flat4to2 s]
   (when [p == b*i]).  Used to discharge MatmulND's [flat3to2 eA %~ rA]. *)
let flat4to2_eq
  (#et:Type) (b i j ll : nat) (p:nat) (_:squash (p == b * i))
  (s : chest4 et b i j ll)
  : Lemma
    (flat3to2 #et #p #j #ll
       (from_seq (l3_batched_row_major p j ll)
          (to_seq (l4_row_major b i j ll) s))
     == flat4to2 s)
  = ()

(* Backward: re-interpreting a (p,j,kk) 3-D matrix [e3] as the 4-D tensor and
   flattening it back recovers [flat3to2 e3].  Used to push MatmulND's
   [flat3to2 eC' %~ matmul] through to the 4-D postcondition. *)
let flat4to2_out
  (#et:Type) (b i j kk : nat) (p:nat) (_:squash (p == b * i))
  (e3 : chest3 et p j kk)
  : Lemma
    (flat4to2 #et #b #i #j #kk
       (from_seq (l4_row_major b i j kk)
          (to_seq (l3_batched_row_major p j kk) e3))
     == flat3to2 e3)
  = let lx = to_seq (l3_batched_row_major p j kk) e3 in
    Kuiper.Tensor.Layout.to_from (l4_row_major b i j kk) lx;
    flat4to2_eq b i j kk p () (from_seq (l4_row_major b i j kk) lx);
    from_to3 (l3_batched_row_major p j kk) e3

(* ----------------------------------------------------------------------- *)
(* DIRECT 4-D matmul spec glue.  We prove a per-entry index lemma relating   *)
(* the row-major flatten [flat4to2] to the 4-D accessor [acc4],     *)
(* then bridge: (i) the approximation [%~] in both directions, and (ii) that *)
(* the DIRECT product [ematmul4] equals the flattened product re-shaped.     *)
(* ----------------------------------------------------------------------- *)

(* [major_on 0]'s index map computes [a * sizeof d + sub.f rest].  Local copy *)
(* of MatmulND's (un-exported) helper; pure layout-algebra fact.             *)
let major_on_zero_f
  (#nn:nat) (kk:nat) (#d:shape nn) (sub:layout_f_for d)
  (a:natlt kk) (rest:abs d)
  : Lemma ((major_on 0 kk sub).f (a, rest) == a * sizeof d + sub.f rest)
  = assert ((major_on 0 kk sub).f (a, rest)
             == major_on_f 0 kk sub (a, rest));
    assert_norm (major_on_f #nn 0 kk #d sub (a, rest)
                   == a * sizeof d + sub.f rest)

(* Layer-2 index identity: collapsing the leading two dims (b,i)->(b*i) of the
   row-major 4-D layout yields the same linear index as the batched 3-D layout
   at row [b_*i+i_].  Analogue of MatmulND's [imap_eq] one level deeper. *)
#push-options "--z3rlimit 40"
let imap_eq4
  (#b #i #j #l : nat)
  (b_:nat{b_<b}) (i_:nat{i_<i}) (j_:nat{j_<j}) (c:nat{c<l})
  : Lemma
    ((l4_row_major b i j l).imap.f (b_,(i_,(j_,(c,()))))
       == (l3_batched_row_major (b*i) j l).imap.f ((b_*i+i_),(j_,(c,()))))
  = let l_l : layout_f_for (l @| INil) = major_on 0 l lunit in
    let l_jl : layout_f_for (j @| l @| INil) = major_on 0 j l_l in
    let l_ijl : layout_f_for (i @| j @| l @| INil) = major_on 0 i l_jl in
    FStar.Math.Lemmas.lemma_mult_le_right i (b_+1) b;
    FStar.Math.Lemmas.distributivity_add_left b_ 1 i;
    let row : natlt (b*i) = b_*i + i_ in
    assert ((l4_row_major b i j l).imap.f (b_,(i_,(j_,(c,()))))
              == (major_on 0 b l_ijl).f (b_,(i_,(j_,(c,())))));
    assert ((l3_batched_row_major (b*i) j l).imap.f (row,(j_,(c,())))
              == (major_on 0 (b*i) l_jl).f (row,(j_,(c,()))));
    major_on_zero_f #3 b #(i @| j @| l @| INil) l_ijl b_ (i_,(j_,(c,())));
    major_on_zero_f #2 i #(j @| l @| INil) l_jl i_ (j_,(c,()));
    major_on_zero_f #2 (b*i) #(j @| l @| INil) l_jl row (j_,(c,()));
    assert (sizeof (i @| j @| l @| INil) == i * sizeof (j @| l @| INil));
    FStar.Math.Lemmas.paren_mul_right b_ i (sizeof (j @| l @| INil));
    FStar.Math.Lemmas.distributivity_add_left (b_*i) i_ (sizeof (j @| l @| INil))
#pop-options

(* Layer-2 content identity: the [flat4to3] collapse places 4-D entry
   [s[b_][i_][j_][c]] at 3-D position [(b_*i+i_)][j_][c].  Analogue of
   MatmulND's [content_ok] for the FIRST-two-dims collapse. *)
let content_ok4to3
  (#et:Type) (#b #i #j #l : nat)
  (s : chest4 et b i j l)
  (b_:nat{b_<b}) (i_:nat{i_<i}) (j_:nat{j_<j}) (c:nat{c<l})
  : Lemma
    (acc3 (flat4to3 s) (b_*i+i_) j_ c == acc4 s b_ i_ j_ c)
  = imap_eq4 #b #i #j #l b_ i_ j_ c;
    let l4 = l4_row_major b i j l in
    let l3 = l3_batched_row_major (b*i) j l in
    let s_flat = to_seq l4 s in
    assert (acc3 (from_seq l3 s_flat) (b_*i+i_) j_ c
              == Seq.index s_flat (l3.imap.f ((b_*i+i_),(j_,(c,())))));
    assert (l3.imap.f ((b_*i+i_),(j_,(c,()))) == l4.imap.f (b_,(i_,(j_,(c,())))));
    assert (Kuiper.Injection.inverse_f l4.imap (l4.imap.f (b_,(i_,(j_,(c,())))))
              == (b_,(i_,(j_,(c,())))))

(* Encode bound: the flattened row index stays in range. *)
let encode3_bound (b i j : nat) (b_:nat{b_<b}) (i_:nat{i_<i}) (j_:nat{j_<j})
  : Lemma ((b_*i+i_)*j+j_ < b*i*j)
  = FStar.Math.Lemmas.distributivity_add_left b_ 1 i;
    FStar.Math.Lemmas.lemma_mult_le_right i (b_+1) b;
    FStar.Math.Lemmas.distributivity_add_left (b_*i+i_) 1 j;
    FStar.Math.Lemmas.lemma_mult_le_right j (b_*i+i_+1) (b*i)

(* Decode: any flat row index [R < b*i*j] splits uniquely into (b_,i_,j_). *)

(* Small, isolated cancellation fact: [a < m*n ==> a/n < m].  Keeping it
   separate gives Z3 a tiny nonlinear query instead of one huge one. *)
let div_bound (a:nat) (n:pos) (m:nat)
  : Lemma (requires a < m * n) (ensures a / n < m)
  = FStar.Math.Lemmas.multiply_fractions a n;             (* n*(a/n) <= a *)
    FStar.Math.Lemmas.swap_mul n (a/n);                   (* n*(a/n) == (a/n)*n *)
    FStar.Math.Lemmas.multiplication_order_lemma (a/n) m n (* a/n>=m <==> (a/n)*n>=m*n *)

#push-options "--z3rlimit 40"
let decode3 (b i j : nat) (r : nat{r < b*i*j})
  : Lemma (ensures (
      let j_ = r % j in let q = r / j in
      let i_ = q % i in let b_ = q / i in
      b_ < b /\ i_ < i /\ j_ < j /\ r == (b_*i+i_)*j+j_))
  = assert (b*i*j > 0);
    assert (j > 0);
    assert (i > 0);
    let j_ = r % j in
    let q = r / j in
    FStar.Math.Lemmas.lemma_mod_lt r j;             (* j_ < j *)
    assert ((b*i)*j == b*i*j);
    div_bound r j (b*i);                             (* q = r/j < b*i *)
    assert (q < b * i);
    let i_ = q % i in
    let b_ = q / i in
    FStar.Math.Lemmas.lemma_mod_lt q i;             (* i_ < i *)
    div_bound q i b;                                 (* b_ = q/i < b *)
    assert (b_ < b);
    (* reconstruction: r = j*q + j_ and q = i*b_ + i_. *)
    FStar.Math.Lemmas.lemma_div_mod r j;            (* r == j*q + j_ *)
    FStar.Math.Lemmas.lemma_div_mod q i;            (* q == i*b_ + i_ *)
    FStar.Math.Lemmas.swap_mul i b_;                (* i*b_ == b_*i *)
    assert (b_*i + i_ == q);
    FStar.Math.Lemmas.swap_mul q j;                 (* q*j == j*q *)
    assert (r == (b_*i+i_)*j+j_)
#pop-options

(* THE KEY INDEX LEMMA: the row-major flatten at decoded row index [R] equals
   the 4-D accessor.  Combines the two collapse layers ([content_ok4to3] for
   (b,i)->b*i, then [content_ok] for (b*i,j)->b*i*j). *)
let flat4to2_index
  (#et:Type) (#b #i #j #l : nat)
  (s : chest4 et b i j l)
  (b_:nat{b_<b}) (i_:nat{i_<i}) (j_:nat{j_<j}) (c:nat{c<l})
  (r : natlt (b*i*j))
  (_ : squash (r == (b_*i+i_)*j+j_))
  : Lemma (acc2 (flat4to2 s) r c == acc4 s b_ i_ j_ c)
  = content_ok4to3 s b_ i_ j_ c;
    content_ok (flat4to3 s) (b_*i+i_) j_ c

(* Bridge pointwise (flat 4-index) approximation to whole-chest4 [%~].
   Replaces the deleted [EMatrix4.lemma_approximates_intro]; the [introduce]
   destructures the nested-tuple [abs] index so SMT can connect it to the
   flat [acc4] hypotheses. *)
let lemma_approximates_intro4
  (#et:Type) {| scalar et, real_like et |} (#d0 #d1 #d2 #d3 : nat)
  (m1 : chest4 et d0 d1 d2 d3) (m2 : chest4 real d0 d1 d2 d3)
  : Lemma (requires forall (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3).
                      acc4 m1 i j k l %~ acc4 m2 i j k l)
          (ensures m1 %~ m2)
  = introduce forall (idx : abs (d0 @| d1 @| d2 @| d3 @| INil)). acc m1 idx %~ acc m2 idx
    with (let (i, (j, (k, (l, ())))) = idx in
          assert (acc4 m1 i j k l %~ acc4 m2 i j k l))

(* L1 forward (flat ==> 4D approximation): used for the postcondition. *)
let approx4_of_flat
  (#et:Type) {| scalar et, real_like et |} (#b #i #j #l:nat)
  (m1 : chest4 et b i j l) (m2 : chest4 real b i j l)
  : Lemma (requires flat4to2 m1 %~ flat4to2 m2)
          (ensures m1 %~ m2)
  = let aux (b_:natlt b) (i_:natlt i) (j_:natlt j) (c:natlt l)
      : Lemma (acc4 m1 b_ i_ j_ c %~ acc4 m2 b_ i_ j_ c)
      = encode3_bound b i j b_ i_ j_;
        let r : natlt (b*i*j) = (b_*i+i_)*j+j_ in
        flat4to2_index m1 b_ i_ j_ c r ();
        flat4to2_index m2 b_ i_ j_ c r ()
    in
    Classical.forall_intro_4 aux;
    lemma_approximates_intro4 m1 m2

(* L1 backward (4D ==> flat approximation): used for the precondition. *)
let flat_of_approx4
  (#et:Type) {| scalar et, real_like et |} (#b #i #j #l:nat)
  (m1 : chest4 et b i j l) (m2 : chest4 real b i j l)
  : Lemma (requires m1 %~ m2)
          (ensures flat4to2 m1 %~ flat4to2 m2)
  = let aux (r:natlt (b*i*j)) (c:natlt l)
      : Lemma (acc2 (flat4to2 m1) r c %~ acc2 (flat4to2 m2) r c)
      = decode3 b i j r;
        let j_ = r % j in let q = r / j in let i_ = q % i in let b_ = q / i in
        flat4to2_index m1 b_ i_ j_ c r ();
        flat4to2_index m2 b_ i_ j_ c r ()
    in
    Classical.forall_intro_2 aux

(* L2: the DIRECT product commutes with the flatten.  This is what lets the
   implementation reuse the flattened MatmulND result for the 4-D postcond. *)
#push-options "--z3rlimit 40"
let ematmul4_flat_lemma
  (#b #i #j #l #k : nat)
  (a : chest4 real b i j l)
  (bm : EMatrix.chest2 real l k)
  : Lemma (flat4to2 (ematmul4 a bm) == MS.matmul (flat4to2 a) bm)
  = let lhs = flat4to2 (ematmul4 a bm) in
    let rhs = MS.matmul (flat4to2 a) bm in
    let aux (r:natlt (b*i*j)) (c:natlt k)
      : Lemma (acc2 lhs r c == acc2 rhs r c)
      = decode3 b i j r;
        let j_ = r % j in let q = r / j in let i_ = q % i in let b_ = q / i in
        flat4to2_index (ematmul4 a bm) b_ i_ j_ c r ();
        MS.lemma_matmul_index (row_pair_slice a b_ i_) bm j_ c;
        MS.lemma_matmul_index (flat4to2 a) bm r c;
        let praux (n:natlt l)
          : Lemma (acc2 (row_pair_slice a b_ i_) j_ n
                     == acc2 (flat4to2 a) r n)
          = flat4to2_index a b_ i_ j_ n r ()
        in
        Classical.forall_intro praux;
        MS.__gmatmul_single_congr zero mul add
          (row_pair_slice a b_ i_) bm (flat4to2 a) bm
          j_ c r c l
    in
    Classical.forall_intro_2 aux;
    EMatrix.lemma_equal_intro lhs rhs;
    Kuiper.Chest.ext lhs rhs
#pop-options

(* ----------------------------------------------------------------------- *)
(* A located variant of [Array4.pts_to_ref]: extract [SZ.fits] for the      *)
(* layout size while the array lives under [on loc (...)].  Mirror of        *)
(* [Array3.pts_to_ref_located].                                            *)
(* ----------------------------------------------------------------------- *)

ghost
fn pts_to_ref_located4
  (#et:Type) (#d0 #d1 #d2 #d3:nat) (#l:layout4 d0 d1 d2 d3)
  (a : array4 et l)
  (#loc:_)
  (#f:perm) (#s:erased (chest4 et d0 d1 d2 d3))
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (tlayout_ulen l))
{
  ghost_impersonate loc
    (on loc (a |-> Frac f s))
    (on loc (a |-> Frac f s) ** pure (SZ.fits (tlayout_ulen l)))
    fn () {
      on_elim _;
      tensor_pts_to_ref a;
      on_intro (a |-> Frac f s);
    }
}

(* ----------------------------------------------------------------------- *)
(* Forward reshape: re-interpret the row-major (b,i,j,ll) Array4 as a        *)
(* row-major (b*i, j, ll) Array3 over the SAME buffer (no data movement).   *)
(* Direct analogue of MatmulND's [reshape3to2].                             *)
(* ----------------------------------------------------------------------- *)

ghost
fn reshape4to3
  (#et:Type)
  (#b #i #j #ll : nat)
  (p : nat)
  (#_ : squash (p == b * i))
  (a4 : array4 et (l4_row_major b i j ll))
  (#s4 : chest4 et b i j ll)
  (#f : perm)
  requires
    a4 |-> Frac f s4
  ensures
    from_array (l3_batched_row_major p j ll) (core a4)
      |-> Frac f (from_seq (l3_batched_row_major p j ll)
                     (to_seq (l4_row_major b i j ll) s4))
{
  tensor_concr a4;
  tensor_abs' (l3_batched_row_major p j ll) (core a4)
}

(* ----------------------------------------------------------------------- *)
(* Backward reshape: re-interpret the flattened (b*i,j,ll) Array3 view back  *)
(* as the (b,i,j,ll) Array4 tensor.  Mirror of MatmulND's [reshape2to3].    *)
(* ----------------------------------------------------------------------- *)

ghost
fn reshape3to4
  (#et:Type)
  (#b #i #j #ll : nat)
  (p : nat)
  (#_ : squash (p == b * i))
  (a4 : array4 et (l4_row_major b i j ll))
  (#s4 : chest4 et b i j ll)
  (#f : perm)
  requires
    from_array (l3_batched_row_major p j ll) (core a4)
      |-> Frac f (from_seq (l3_batched_row_major p j ll)
                     (to_seq (l4_row_major b i j ll) s4))
  ensures
    a4 |-> Frac f s4
{
  tensor_concr (from_array (l3_batched_row_major p j ll) (core a4));
  rewrite
    (core (from_array (l3_batched_row_major p j ll) (core a4))
      |-> Frac f (to_seq (l3_batched_row_major p j ll)
                    (from_seq (l3_batched_row_major p j ll)
                       (to_seq (l4_row_major b i j ll) s4))))
  as
    (core a4 |-> Frac f (to_seq (l4_row_major b i j ll) s4));
  tensor_abs (l4_row_major b i j ll) (core a4) #f #s4;
  rewrite
    (from_array (l4_row_major b i j ll) (core a4) |-> Frac f s4)
  as
    (a4 |-> Frac f s4);
}

(* Variant of [reshape3to4] for an output whose flat contents are only KNOWN
   to equal the [from_seq] shape (e.g. the MatmulND result tensor); the
   supplied equality lets Pulse match the [pts_to] value before delegating. *)
ghost
fn reshape3to4_eq
  (#et:Type)
  (#b #i #j #ll : nat)
  (p : nat)
  (#_ : squash (p == b * i))
  (a4 : array4 et (l4_row_major b i j ll))
  (#s4 : chest4 et b i j ll)
  (#f : perm)
  (#e : chest3 et p j ll)
  (#_ : squash (
     e == from_seq (l3_batched_row_major p j ll)
            (to_seq (l4_row_major b i j ll) s4)))
  requires
    from_array (l3_batched_row_major p j ll) (core a4) |-> Frac f e
  ensures
    a4 |-> Frac f s4
{
  rewrite
    (from_array (l3_batched_row_major p j ll) (core a4) |-> Frac f e)
  as
    (from_array (l3_batched_row_major p j ll) (core a4)
      |-> Frac f (from_seq (l3_batched_row_major p j ll)
                     (to_seq (l4_row_major b i j ll) s4)));
  reshape3to4 p a4 #s4 #f;
}

(* ----------------------------------------------------------------------- *)
(* The kernel.                                                              *)
(* ----------------------------------------------------------------------- *)

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn matmul4d
  (#t:Type0) {| floating t, real_like t, floating_real_like t |}
  (b i j l k : szp)
  (gA : array4 t (l4_row_major b i j l) { is_global gA })
  (gB : array2 t (l2_row_major l k)     { is_global gB })
  (gC : array4 t (l4_row_major b i j k) { is_global gC })
  (rA : chest4 real b i j l)
  (rB : EMatrix.chest2 real l k)
  (#eA : chest4 t b i j l)
  (#eB : EMatrix.chest2 t l k)
  (#eC : chest4 t b i j k)
  (#fA #fB : perm)
  requires
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB) **
    pure (eA %~ rA) **
    pure (eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB) **
    (exists* (eC' : chest4 t b i j k).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ ematmul4 rA rB))
{
  (* Derive [SZ.fits] for the GPU array layout sizes from their [pts_to]:
     gA gives fits(b*i*j*l), gC gives fits(b*i*j*k).  These justify forming
     the machine products below. *)
  pts_to_ref_located4 gA;
  pts_to_ref_located4 gC;
  assert pure (SZ.fits (SZ.v b * SZ.v i));

  (* Runtime self-protection: instead of relying on the C++ bridge to pre-check
     the index/launch bounds, [dguard] aborts here if any product would overflow
     the uint32 ABI.  Each guard is a division-style test computed BEFORE the
     corresponding machine multiply, so it fires even when the C product would
     silently wrap.  The [pts_to]-derived [SZ.fits] facts discharge the proof;
     the guards make that ABI assumption true at runtime. *)
  let maxu32 : sz = SZ.uint_to_t 4294967295;
  dguard (i <=^ (maxu32 /^ b));            (* b*i fits *)
  let bi : szp = b *^ i;
  assert pure (SZ.v bi == SZ.v b * SZ.v i);
  dguard (j <=^ (maxu32 /^ bi));           (* b*i*j fits *)
  let bij : sz = bi *^ j;
  assert pure (SZ.v bij == SZ.v b * SZ.v i * SZ.v j);
  dguard (l <=^ (maxu32 /^ bij));          (* b*i*j*l fits (gA) *)
  dguard (k <=^ (maxu32 /^ bij));          (* b*i*j*k fits (gC) *)
  let bijk : sz = bij *^ k;
  assert pure (SZ.v bijk == SZ.v b * SZ.v i * SZ.v j * SZ.v k);
  dguard (k <=^ (maxu32 /^ l));            (* l*k fits (gB) *)

  (* Establish the kernel launch bound [size_req] at runtime rather than as a
     precondition: (b*i*j)*k <= max_blocks*max_threads. *)
  let bound : sz = max_blocks *^ max_threads;
  assert pure (SZ.v bound == max_blocks * max_threads);
  dguard (bijk <=^ bound);
  assert pure (K.size_req (SZ.v bi * SZ.v j) (SZ.v k) (SZ.v l));

  let pbi : erased nat = SZ.v bi;

  (* The flattened (b*i*j, l) real operand: the spec-level mirror of the buffer
     re-interpretation, defined directly from the 4-D real model [rA]. *)
  let rA_flat : EMatrix.chest2 real (SZ.v b * SZ.v i * SZ.v j) l = flat4to2 rA;

  (* 1. Forward-reshape gA : (b,i,j,l) -> 3-D view (b*i,j,l) over same buffer. *)
  map_loc gpu_loc (fun () -> reshape4to3 pbi gA);
  (* 2. Forward-reshape gC : (b,i,j,k) -> 3-D view (b*i,j,k). *)
  map_loc gpu_loc (fun () -> reshape4to3 pbi gC);

  (* Bridge: MatmulND wants [flat3to2 eA3 %~ rA_flat].  [flat4to2_eq] gives
     [flat3to2 eA3 == flat4to2 eA]; [flat_of_approx4] transfers the new 4-D
     hypothesis [eA %~ rA] to [flat4to2 eA %~ flat4to2 rA == rA_flat]. *)
  flat4to2_eq #t (SZ.v b) (SZ.v i) (SZ.v j) (SZ.v l) pbi () eA;
  flat_of_approx4 #t #_ #_ #(SZ.v b) #(SZ.v i) #(SZ.v j) #(SZ.v l) eA rA;

  (* 3. Run the verified ND matmul on the flattened (b*i,j,l) @ (l,k) operands. *)
  matmul_nd #t bi j l k
    (from_array (l3_batched_row_major pbi j l) (core gA))
    gB
    (from_array (l3_batched_row_major pbi j k) (core gC))
    rA_flat rB;
  with eC3.
    assert on gpu_loc
      (from_array (l3_batched_row_major pbi j k) (core gC) |-> eC3);
  (* context hyp:  flat3to2 eC3 %~ MS.matmul rA_flat rB *)

  (* 4. Backward-reshape gA view -> gA (recover gA |-> eA). *)
  map_loc gpu_loc (fun () -> reshape3to4 pbi gA);

  (* 5. Backward-reshape gC view -> gC, and prove the functional postcondition.
        eC4 re-interprets the (b*i,j,k) result as the (b,i,j,k) tensor. *)
  let eC4 : chest4 t b i j k =
    from_seq (l4_row_major b i j k)
      (to_seq (l3_batched_row_major pbi j k) eC3);
  flat4to2_out #t (SZ.v b) (SZ.v i) (SZ.v j) (SZ.v k) pbi () eC3;
  (* [flat4to2 eC4 == flat3to2 eC3 %~ MS.matmul rA_flat rB]. *)
  assert pure (flat4to2 eC4 %~ MS.matmul rA_flat rB);
  (* DIRECT product commutes with the flatten:
     [flat4to2 (ematmul4 rA rB) == MS.matmul (flat4to2 rA) rB == MS.matmul rA_flat rB],
     so [flat4to2 eC4 %~ flat4to2 (ematmul4 rA rB)]; then [approx4_of_flat] lifts
     that to the DIRECT 4-D postcondition [eC4 %~ ematmul4 rA rB]. *)
  ematmul4_flat_lemma #(SZ.v b) #(SZ.v i) #(SZ.v j) #(SZ.v l) #(SZ.v k) rA rB;
  approx4_of_flat #t #_ #_ #(SZ.v b) #(SZ.v i) #(SZ.v j) #(SZ.v k) eC4 (ematmul4 rA rB);
  assert pure (eC4 %~ ematmul4 rA rB);
  (* [eC3 == from_seq (l3 pbi) (to_seq (l4) eC4)] discharges the reshape3to4_eq
     obligation: to_seq (l4) eC4 == to_seq (l3 pbi) eC3 (A4.to_from), then
     from_seq (l3 pbi) (to_seq (l3 pbi) eC3) == eC3 (from_to3). *)
  from_to3 (l3_batched_row_major pbi j k) eC3;
  map_loc gpu_loc (fun () -> reshape3to4_eq pbi gC #eC4 #_ #eC3);
  ()
}
#pop-options

let matmul4d_f32 = matmul4d
