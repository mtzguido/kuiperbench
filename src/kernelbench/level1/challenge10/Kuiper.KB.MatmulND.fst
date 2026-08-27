module Kuiper.KB.MatmulND

(* Implementation: see Kuiper.KB.MatmulND.fsti for the design overview.

   The kernel re-interprets the SAME GPU buffer between the 3-D (N,M,K) view
   and the 2-D (N*M,K) view using ghost [lower]/[raise] (no data movement),
   runs the layout-polymorphic Naive3 GEMM on the flattened operands, and
   re-interprets the (N*M,L) result back to the (N,M,L) tensor.

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
module Klas3 = Klas.GEMM.Naive3
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Bijection

(* Bridges between flat array2 ownership and its zero-cost rank-2 tensor view,
   required to feed the (tensor-based) Naive3 GEMM API while keeping the rest
   of the proof array2-based.  Mirror of
   [Kuiper.KB.GemmDivSumScale.bridge_fwd/bwd].  Both are [ghost] and
   [as_tensor] is [inline_for_extraction noextract], so no extracted CUDA
   changes. *)
ghost
fn bridge_fwd
  (#et : Type0) (#rows #cols : nat) (#lay : layout2 rows cols)
  (a : array2 et lay) (#f : perm) (#s : EMatrix.chest2 et rows cols)
  preserves on gpu_loc (a |-> Frac f s)
{
  rewrite (on gpu_loc (a |-> Frac f s))
       as (on gpu_loc (a |-> Frac f s));
}

ghost
fn bridge_bwd
  (#et : Type0) (#rows #cols : nat) (#lay : layout2 rows cols)
  (a : array2 et lay) (#f : perm) (#s : EMatrix.chest2 et rows cols)
  preserves on gpu_loc (a |-> Frac f s)
{
  rewrite (on gpu_loc (a |-> Frac f s))
       as (on gpu_loc (a |-> Frac f s));
}


(* ----------------------------------------------------------------------- *)
(* Reshape lemmas (moved from Scratch.Reshape — these verify clean and      *)
(* document that the flatten is exactly the batched row-major semantics).   *)
(* ----------------------------------------------------------------------- *)

let major_on_zero_f
  (#nn:nat) (kk:nat) (#d:shape nn) (sub:layout_f_for d)
  (a:natlt kk) (rest:abs d)
  : Lemma ((major_on 0 kk sub).f (a, rest) == a * sizeof d + sub.f rest)
  = assert ((major_on 0 kk sub).f (a, rest)
             == major_on_f 0 kk sub (a, rest));
    assert_norm (major_on_f #nn 0 kk #d sub (a, rest)
                   == a * sizeof d + sub.f rest)

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

let content_ok
  (#et:Type) (#n #m #k : nat)
  (s3 : chest3 et n m k)
  (i:nat{i<n}) (j:nat{j<m}) (c:nat{c<k})
  : Lemma
    (acc2
       (from_seq (l2_row_major (n*m) k)
          (to_seq (l3_batched_row_major n m k) s3))
       (i*m+j) c
     == acc3 s3 i j c)
  = imap_eq #n #m #k i j c;
    let l3 = l3_batched_row_major n m k in
    let l2 = l2_row_major (n*m) k in
    let s_flat = to_seq l3 s3 in
    assert (acc2 (from_seq l2 s_flat) (i*m+j) c
              == Seq.index s_flat (l2.imap.f ((i*m+j),(c,()))));
    assert (l2.imap.f ((i*m+j),(c,())) == l3.imap.f (i,(j,(c,()))));
    assert (Kuiper.Injection.inverse_f l3.imap (l3.imap.f (i,(j,(c,())))) == (i,(j,(c,()))))

(* ----------------------------------------------------------------------- *)
(* [from_seq (to_seq s) == s] — the "round-trip the other way" that is NOT  *)
(* in the Array2 library (which only provides [to_seq (from_seq x) == x]).  *)
(* Proof by matrix extensionality + injectivity of the layout map.          *)
(* ----------------------------------------------------------------------- *)

let from_to2
  (#et:Type) (#m #n:nat)
  (l : full_layout2 m n)
  (s : EMatrix.chest2 et m n)
  : Lemma (ensures from_seq l (to_seq l s) == s)
  = let lhs = from_seq l (to_seq l s) in
    let aux (i:natlt m) (j:natlt n)
      : Lemma (acc2 lhs i j == acc2 s i j)
      = inverse_lem l.imap (l.imap.f (i,(j,())));
        l.imap.is_inj (inverse_f l.imap (l.imap.f (i,(j,())))) (i,(j,()))
    in
    Classical.forall_intro_2 aux;
    EMatrix.lemma_equal_intro lhs s;
    Kuiper.Chest.ext lhs s

(* Per-entry bridge between the runtime product and its mathematical value.
   The GEMM output [eGemm] is typed at [pnm == n*m]; this lemma
   shows that the (n,m,l)-tensor re-interpretation, flattened back via [flat3to2]
   (whose row dimension is the NAT product [n*m]), has the SAME entries as
   [eGemm].  Reasoning is purely at the scalar [acc2] level (no [%~] typeclass),
   so the propositional equality [pnm == n*m] suffices (no type transport). *)
#push-options "--z3rlimit 50"
let entry_eq
  (#t:Type) (n m k l : nat) (pnm:nat) (_:squash (pnm == n * m))
  (eGemm : EMatrix.chest2 t pnm l)
  (i:natlt (n*m)) (j:natlt l)
  : Lemma
    (acc2
       (flat3to2 #t #n #m #l
          (from_seq (l3_batched_row_major n m l)
             (to_seq (l2_row_major pnm l) eGemm)))
       i j
     == acc2 eGemm i j)
  = let l2p = l2_row_major pnm l in
    let l2n = l2_row_major (n*m) l in
    (* l2n == l2p by congruence of [l2_row_major] on [pnm == n*m]. *)
    assert (l2n == l2p);
    inverse_lem l2p.imap (l2p.imap.f (i,(j,())));
    l2p.imap.is_inj (inverse_f l2p.imap (l2p.imap.f (i,(j,())))) (i,(j,()))
#pop-options

(* ----------------------------------------------------------------------- *)
(* Backward reshape: re-interpret the flattened (n*m,k) Array2 view back as  *)
(* the (n,m,k) Array3 tensor.  Requires the flat contents to be exactly      *)
(* [flat3to2 s3] for some 3-D tensor [s3]; restores [a3 |-> s3].             *)
(* ----------------------------------------------------------------------- *)

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
  (* Re-express the lowered [pts_to]: the array [core (from_array l2 p)]
     reduces to [p == core a3] (A2.lem_from_array_core SMTPat), and the value
     [to_seq l2 (from_seq l2 X)] reduces to [X] (A2.to_from SMTPat). *)
  rewrite
    (core (from_array (l2_row_major p k) (core a3))
      |-> Frac f (to_seq (l2_row_major p k)
                    (from_seq (l2_row_major p k)
                       (to_seq (l3_batched_row_major n m k) s3))))
  as
    (core a3 |-> Frac f (to_seq (l3_batched_row_major n m k) s3));
  tensor_abs (l3_batched_row_major n m k) (core a3) #f #s3;
  (* [from_array l3 (core a3)] reduces to [a3] (A3.lem_core_from_array). *)
  rewrite
    (from_array (l3_batched_row_major n m k) (core a3) |-> Frac f s3)
  as
    (a3 |-> Frac f s3);
}

(* Variant of [reshape2to3] for an output whose flat contents are only KNOWN to
   equal [flat3to2 s3] (e.g. the GEMM result [eGemm]); the supplied equality
   lets Pulse match the [pts_to] value before delegating to [reshape2to3]. *)
ghost
fn reshape2to3_eq
  (#et:Type)
  (#n #m #k : nat)
  (p : nat)
  (#_ : squash (p == n * m))
  (a3 : array3 et (l3_batched_row_major n m k))
  (#s3 : chest3 et n m k)
  (#f : perm)
  (#e : EMatrix.chest2 et p k)
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
(* The kernel.                                                              *)
(* ----------------------------------------------------------------------- *)

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn matmul_nd
  (#t:Type0) {| floating t, real_like t, floating_real_like t |}
  (n m k l : szp)
  (gA : array3 t (l3_batched_row_major n m k) { is_global gA })
  (gB : array2 t (l2_row_major k l)           { is_global gB })
  (gC : array3 t (l3_batched_row_major n m l) { is_global gC })
  (rA : EMatrix.chest2 real (n*m) k)
  (rB : EMatrix.chest2 real k l)
  (#eA : chest3 t n m k)
  (#eB : EMatrix.chest2 t k l)
  (#eC : chest3 t n m l)
  (#fA #fB : perm)
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (K.size_req (n*m) l k) **
    pure (flat3to2 eA %~ rA) **
    pure (eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest3 t n m l).
      on gpu_loc (gC |-> eC') **
      pure (flat3to2 eC' %~ MS.matmul rA rB))
{
  (* Derive [SZ.fits (n*m)] from [gA]'s layout size [sizeof l3 == n*m*k]. *)
  tensor_pts_to_ref_located gA;
  assert pure (SZ.fits (SZ.v n * SZ.v m));
  let nm : szp = n *^ m;
  assert pure (SZ.v nm == SZ.v n * SZ.v m);
  let pnm : erased nat = SZ.v nm;

  (* 1. Forward-reshape gA : (n,m,k) -> 2-D view gA2 : (n*m,k) over same buffer. *)
  map_loc gpu_loc (fun () -> reshape3to2 pnm gA);
  (* on gpu_loc (gA2 |-> Frac fA (flat3to2 eA)) where
        gA2 == from_array (l2_row_major pnm k) (core gA) *)

  (* 2. Forward-reshape gC : (n,m,l) -> 2-D view gC2 : (n*m,l). *)
  map_loc gpu_loc (fun () -> reshape3to2 pnm gC);
  (* on gpu_loc (gC2 |-> Frac 1.0 (flat3to2 eC)) *)

  (* 3. Run the layout-polymorphic Naive3 GEMM on the flattened operands.
        Preserves gA2, gB; turns gC2 into the matmul result.
        Bridge Array2 ownership into tensor ownership for the GEMM API. *)
  bridge_fwd (from_array (l2_row_major pnm k) (core gA));
  bridge_fwd gB;
  bridge_fwd (from_array (l2_row_major pnm l) (core gC));
  Klas3.spec t l2_row_major l2_row_major l2_row_major nm l k
    ((from_array (l2_row_major pnm k) (core gA)))
    (gB)
    ((from_array (l2_row_major pnm l) (core gC)))
    rA rB;
  with eGemm.
    assert on gpu_loc
      ((from_array (l2_row_major pnm l) (core gC)) |-> eGemm);
  bridge_bwd (from_array (l2_row_major pnm k) (core gA));
  bridge_bwd gB;
  bridge_bwd (from_array (l2_row_major pnm l) (core gC));
  (* context hyp:  eGemm %~ MS.matmul rA rB   (at the [pnm] row dimension) *)

  (* 4. Backward-reshape gA2 -> gA (recover gA |-> eA). *)
  map_loc gpu_loc (fun () -> reshape2to3 pnm gA);

  (* 5. Backward-reshape gC2 -> gC, and prove the functional postcondition.
        eC' re-interprets the (pnm,l) GEMM result as the (n,m,l) tensor. *)
  let eC' : chest3 t n m l =
    from_seq (l3_batched_row_major n m l)
      (to_seq (l2_row_major pnm l) eGemm);
  (* [flat3to2 eC'] (row dim [n*m]) has the same entries as [eGemm] (row dim
     [pnm]); combined with [eGemm %~ matmul] and [lemma_matmul_index] (which
     makes [acc2 (matmul ..)] independent of the row-dim ascription), this gives
     the elementwise [%~] postcondition. *)
  Classical.forall_intro_2 (entry_eq n m k l pnm () eGemm);
  assert pure (flat3to2 eC' %~ MS.matmul rA rB);
  (* [from_seq (l2 pnm) (to_seq (l2 pnm) eGemm) == eGemm] (from_to2), combined
     with [to_seq3 l3 eC' == to_seq (l2 pnm) eGemm] (A3.to_from), discharges the
     [reshape2to3_eq] obligation [eGemm == from_seq (l2 pnm) (to_seq3 l3 eC')]. *)
  from_to2 (l2_row_major pnm l) eGemm;
  map_loc gpu_loc (fun () -> reshape2to3_eq pnm gC #eC' #_ #eGemm);
  (* on gpu_loc (gC |-> eC') ** pure (flat3to2 eC' %~ MS.matmul rA rB) *)
  ()
}
#pop-options

let matmul_nd_f32 = matmul_nd
