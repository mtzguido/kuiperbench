module Kuiper.KB.MatmulND

(* KernelBench L1 #10 — (N,M,K) tensor @ (K,L) matrix -> (N,M,L).
   The SAME matrix B multiplies every outer (N,M) slice.

   Instead of an UNVERIFIED host-side flatten (numel/shared reshape in C++),
   we flatten the row-major Array3 (N,M,K) into a row-major Array2 (N*M,K)
   INSIDE the verified kernel: this is a pure ghost re-interpretation of the
   SAME GPU buffer (no data movement), justified by the [reshape3to2] /
   [content_ok] lemmas below.  We then run the existing layout-polymorphic
   Naive3 GEMM on the (N*M,K) @ (K,L) product and re-interpret the (N*M,L)
   output back as the (N,M,L) tensor.

   Functional spec: the flattened output equals the matmul of the flattened
   input with B, in the real-number approximation [%~] of the Kahan-summed
   Naive3 kernel.  [content_ok] documents that this flatten is exactly the
   batched semantics: flattened row [i*M+j] holds A3[i][j][:].

   Zero assume · zero magic · zero admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor.Layout.Alg { l2_row_major, l3_batched_row_major }
open Kuiper.Tensor
module EMatrix = Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM
module K = Kuiper.Kernel.GEMM.Naive3

(* Verified flatten of a row-major (a,b,c) tensor into a row-major (a*b,c)
   matrix.  This is the spec-level mirror of the ghost buffer re-interpretation
   performed by the kernel ([reshape3to2]). *)
inline_for_extraction noextract
let flat3to2 (#et:Type) (#a #b #c:nat) (s : chest3 et a b c)
  : EMatrix.chest2 et (a*b) c
  = from_seq (l2_row_major (a*b) c)
       (to_seq (l3_batched_row_major a b c) s)

(* Per-entry semantics of the flatten: the flattened row [i*m+j] holds entry
   [s3[i][j][:]].  Exposed so the 4-D variant (challenge 11) can chain a second
   leading-dim collapse on top of this one. *)
val content_ok
  (#et:Type) (#n #m #k : nat)
  (s3 : chest3 et n m k)
  (i:nat{i<n}) (j:nat{j<m}) (c:nat{c<k})
  : Lemma
    (acc2
       (from_seq (l2_row_major (n*m) k)
          (to_seq (l3_batched_row_major n m k) s3))
       (i*m+j) c
     == acc3 s3 i j c)

(* Generic (type-polymorphic) entry — exposed so other KernelBench modules
   (e.g. the 4-D variant in challenge 11) can reuse the verified flatten+GEMM
   without re-deriving it.  [inline_for_extraction noextract]: it generates no
   code on its own; only monomorphic [let]-bindings like [matmul_nd_f32] do. *)
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

fn matmul_nd_f32
  (n m k l : szp)
  (gA : array3 f32 (l3_batched_row_major n m k) { is_global gA })
  (gB : array2 f32 (l2_row_major k l)           { is_global gB })
  (gC : array3 f32 (l3_batched_row_major n m l) { is_global gC })
  (rA : EMatrix.chest2 real (n*m) k)
  (rB : EMatrix.chest2 real k l)
  (#eA : chest3 f32 n m k)
  (#eB : EMatrix.chest2 f32 k l)
  (#eC : chest3 f32 n m l)
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
    (exists* (eC' : chest3 f32 n m l).
      on gpu_loc (gC |-> eC') **
      pure (flat3to2 eC' %~ MS.matmul rA rB))
