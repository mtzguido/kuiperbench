module Kuiper.KB.TransposedGEMM

(* Verified transposed-operand entry points for KernelBench L1 #16/#17/#18.

   [Klas.GEMM.Naive3] is supplied by the pinned Kuiper package.  Its generic
   [spec] supports arbitrary concrete layouts, but the package only exports
   row/row/row and col/col/col monomorphisations.  These KuiperBench-specific
   entries use the package implementation while keeping the row-to-column
   reinterpretation, and therefore the logical transpose, inside the proof. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.EMatrix { mtranspose }
open Kuiper.Ghost.TensorTranspose
module MS = Kuiper.Spec.GEMM
module K = Kuiper.Kernel.GEMM.Naive3
module Klas3 = Klas.GEMM.Naive3

let mtranspose_involutive (#et:Type) (#r #c : nat) (m : chest2 et r c)
  : Lemma (ensures mtranspose (mtranspose m) == m)
          [SMTPat (mtranspose (mtranspose m))]
  = assert (Kuiper.Chest.equal (mtranspose (mtranspose m)) m)

let lemma_mtranspose_approx
  (#et:Type) {| scalar et |} {| real_like et |} (#r #c : nat)
  (e : chest2 et r c) (rr : chest2 real r c)
  : Lemma (requires e %~ rr) (ensures mtranspose e %~ mtranspose rr)
  = ()

inline_for_extraction noextract
fn spec_atb
  (et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m n k : szp)
  (gA : tensor et (l2_row_major k m) { is_global gA })
  (gB : tensor et (l2_row_major k n) { is_global gB })
  (gC : tensor et (l2_row_major m n) { is_global gC })
  (rA : chest2 real k m)
  (rB : chest2 real k n)
  (#eA : chest2 et k m)
  (#eB : chest2 et k n)
  (#eC : chest2 et m n)
  preserves
    cpu ** on gpu_loc (gA |-> eA ** gB |-> eB)
  requires
    pure (K.size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul (mtranspose rA) rB)
{
  lemma_mtranspose_approx eA rA;
  map_loc gpu_loc (fun () -> ghost_transpose1 gA);
  Klas3.spec et l2_col_major l2_row_major l2_row_major m n k
    (row2col gA) gB gC (mtranspose rA) rB;
  map_loc gpu_loc (fun () -> ghost_transpose1_back gA);
}

inline_for_extraction noextract
fn spec_abt
  (et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m n k : szp)
  (gA : tensor et (l2_row_major m k) { is_global gA })
  (gB : tensor et (l2_row_major n k) { is_global gB })
  (gC : tensor et (l2_row_major m n) { is_global gC })
  (rA : chest2 real m k)
  (rB : chest2 real n k)
  (#eA : chest2 et m k)
  (#eB : chest2 et n k)
  (#eC : chest2 et m n)
  preserves
    cpu ** on gpu_loc (gA |-> eA ** gB |-> eB)
  requires
    pure (K.size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA (mtranspose rB))
{
  lemma_mtranspose_approx eB rB;
  map_loc gpu_loc (fun () -> ghost_transpose1 gB);
  Klas3.spec et l2_row_major l2_col_major l2_row_major m n k
    gA (row2col gB) gC rA (mtranspose rB);
  map_loc gpu_loc (fun () -> ghost_transpose1_back gB);
}

inline_for_extraction noextract
fn spec_atbt
  (et : Type0) {| floating et, real_like et, floating_real_like et |}
  (m n k : szp)
  (gA : tensor et (l2_row_major k m) { is_global gA })
  (gB : tensor et (l2_row_major n k) { is_global gB })
  (gC : tensor et (l2_row_major m n) { is_global gC })
  (rA : chest2 real k m)
  (rB : chest2 real n k)
  (#eA : chest2 et k m)
  (#eB : chest2 et n k)
  (#eC : chest2 et m n)
  preserves
    cpu ** on gpu_loc (gA |-> eA ** gB |-> eB)
  requires
    pure (K.size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul (mtranspose rA) (mtranspose rB))
{
  lemma_mtranspose_approx eA rA;
  lemma_mtranspose_approx eB rB;
  map_loc gpu_loc (fun () -> ghost_transpose1 gA);
  map_loc gpu_loc (fun () -> ghost_transpose1 gB);
  Klas3.spec et l2_col_major l2_col_major l2_row_major m n k
    (row2col gA) (row2col gB) gC (mtranspose rA) (mtranspose rB);
  map_loc gpu_loc (fun () -> ghost_transpose1_back gA);
  map_loc gpu_loc (fun () -> ghost_transpose1_back gB);
}

let matmul_f32_atb = spec_atb f32
let matmul_f32_abt = spec_abt f32
let matmul_f32_atbt = spec_atbt f32
