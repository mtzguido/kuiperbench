module Kuiper.Kernel.BiasAdd

(* Generic broadcast bias-add kernel.

   For an [m x n] row-major matrix [C] (held as an [array2]) and a
   length-[n] vector [bias], computes the flattened result

       y[tid] = add (C[tid / n, tid % n]) (bias[tid % n])

   one output element per thread ([m * n] threads).  In matrix terms,
   viewing [y] row-major as an [m x n] matrix,

       y[i, j] = add (C[i, j]) (bias[j]).

   The matrix [C] and the [bias] vector are read-only (held with
   fractional permissions [fc] / [fb], following the [gbias] read in
   [Kuiper.Kernel.Conv2D.Naive]).  The output [y] is a fresh length
   [m * n] array.  This is the elementwise "broadcast bias" used by the
   Linear / Gemm+bias family of KernelBench challenges: after a GEMM
   produces the [m x n] result as an [Array2], this kernel adds the
   per-column bias and flattens to an [Array1] suitable for the
   downstream pointwise activation maps.

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module Seq = FStar.Seq
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT

(* Per-thread functional postcondition: decode [tid] into (i, j) via
   row-major unflatten, read [C[i,j]] and [bias[j]], and add. *)
let bias_add_at
  (#t:Type0) {| scalar t |}
  (m n : nat)
  (eC : EM.chest2 t m n)
  (sbias : chest1 t n)
  (tid : nat{tid < m * n})
  : GTot t
  = let i : natlt m = tid / n in
    let j : natlt n = tid % n in
    add (acc2 eC i j) (acc1 sbias j)

(* For the matrix index [(i, j)], the flat result at [i*n+j] is exactly
   [add C[i,j] bias[j]].  Lets callers phrase the spec in [chest2] form. *)
val bias_add_at_ij
  (#t:Type0) {| scalar t |}
  (m n : nat)
  (eC : EM.chest2 t m n)
  (sbias : chest1 t n)
  (i : natlt m) (j : natlt n)
  : Lemma (bias_add_at m n eC sbias (i * n + j) == add (acc2 eC i j) (acc1 sbias j))

(* The size_t precondition required from callers. *)
unfold
let bias_add_size_req (m n : nat) : prop
  = SZ.fits (m * n) /\
    m * n <= max_blocks * max_threads

(* Generic, layout-polymorphic bias-add kernel: one launch. *)
inline_for_extraction noextract
fn bias_add_gpu
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lC : layout2 (SZ.v m) (SZ.v n)) {| ctlayout lC |}
  (#lbias : layout1 (SZ.v n)) {| ctlayout lbias |}
  (#ly : layout1 (SZ.v m * SZ.v n)) {| ctlayout ly |}
  (gC : array2 t lC)
  (gbias : array1 t lbias)
  (gy : array1 t ly)
  (#eC : erased (EM.chest2 t (SZ.v m) (SZ.v n)))
  (#sbias : erased (chest1 t (SZ.v n)))
  (#sy0 : erased (chest1 t (SZ.v m * SZ.v n)))
  (#fc #fb : perm)
  preserves cpu
  requires
    on gpu_loc (gC |-> Frac fc eC) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gC /\ is_global gbias /\ is_global gy /\
          bias_add_size_req (SZ.v m) (SZ.v n))
  ensures
    on gpu_loc (gC |-> Frac fc eC) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 t (SZ.v m * SZ.v n)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < SZ.v m * SZ.v n}).
               acc1 sy tid == bias_add_at (SZ.v m) (SZ.v n) eC sbias tid))

(* Canonical-layout monomorphic type abbreviation + f32 instantiation. *)
inline_for_extraction noextract
type bias_add_ty (t:Type0) {| scalar t |} =
  fn (m n : szp)
     (gC : array2 t (l2_row_major (SZ.v m) (SZ.v n)) { is_global gC })
     (gbias : array1 t (l1_forward (SZ.v n)) { is_global gbias })
     (gy : array1 t (l1_forward (SZ.v m * SZ.v n)) { is_global gy })
     (#eC : erased (EM.chest2 t (SZ.v m) (SZ.v n)))
     (#sbias : erased (chest1 t (SZ.v n)))
     (#sy0 : erased (chest1 t (SZ.v m * SZ.v n)))
     (#fc #fb : perm)
     preserves cpu
     requires
       on gpu_loc (gC |-> Frac fc eC) **
       on gpu_loc (gbias |-> Frac fb sbias) **
       on gpu_loc (gy |-> sy0) **
       pure (bias_add_size_req (SZ.v m) (SZ.v n))
     ensures
       on gpu_loc (gC |-> Frac fc eC) **
       on gpu_loc (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 t (SZ.v m * SZ.v n)).
          on gpu_loc (gy |-> sy) **
          pure (forall (tid : nat{tid < SZ.v m * SZ.v n}).
                  acc1 sy tid == bias_add_at (SZ.v m) (SZ.v n) eC sbias tid))

val bias_add_f32 : bias_add_ty f32
