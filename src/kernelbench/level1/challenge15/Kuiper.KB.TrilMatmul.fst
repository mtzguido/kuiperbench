module Kuiper.KB.TrilMatmul

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major, c_l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module Tril = Kuiper.Kernel.Tril
module P = Kuiper.Kernel.GEMM.Naive3

(* Bridges between flat array2 ownership and its zero-cost rank-2 tensor view,
   required to feed the (tensor-based) GEMM API while keeping the rest of the
   proof array2-based.  Mirror of [Kuiper.KB.GemmDivSumScale.bridge_fwd/bwd].
   Both are [ghost] and [as_tensor] is [inline_for_extraction noextract], so
   no extracted CUDA changes. *)
ghost
fn bridge_fwd
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (#f : perm) (#s : EM.chest2 et rows cols)
  requires on gpu_loc (a |-> Frac f s)
  ensures  on gpu_loc (a |-> Frac f s)
{
  rewrite (on gpu_loc (a |-> Frac f s))
       as (on gpu_loc (a |-> Frac f s));
}

ghost
fn bridge_bwd
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (#f : perm) (#s : EM.chest2 et rows cols)
  requires on gpu_loc (a |-> Frac f s)
  ensures  on gpu_loc (a |-> Frac f s)
{
  rewrite (on gpu_loc (a |-> Frac f s))
       as (on gpu_loc (a |-> Frac f s));
}

(* Per-(i,j) discharge of [tril_matmul_post].  After the in-place mask the
   output matrix is [s_tril eC'], so entry [(i, j)] is [acc2 eC' i j] on/below
   the diagonal and exactly [zero] above it.  The GEMM's elementwise
   approximation [eC' %~ rAB] (i.e. [acc2 eC' i j %~ acc2 rAB i j]) gives the
   on/below case; [zero %~ 0.0R] gives the above case.  No flat indices. *)
#push-options "--z3rlimit 50"
let tril_row_aux
  (n : nat)
  (eC' : EM.chest2 f32 n n)
  (rAB : EM.chest2 real n n)
  (h_approx : squash (eC' %~ rAB))
  (i j : natlt n)
  : Lemma
      (ensures
        acc2 (Tril.s_tril eC') i j %~
          (if j <= i then acc2 rAB i j else 0.0R))
  = assert ((zero #f32) %~ 0.0R)
#pop-options

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn tril_matmul_f32_impl
  (n : szp {
     SZ.v n * SZ.v n <= max_blocks * max_threads /\
     SZ.fits (SZ.v n * SZ.v n) })
  (gA : array2 f32 (l2_row_major n n) { is_global gA })
  (gB : array2 f32 (l2_row_major n n) { is_global gB })
  (y  : array2 f32 (l2_row_major n n) { is_global y  })
  (#sA : EM.chest2 f32 n n)
  (#sB : EM.chest2 f32 n n)
  (#sy : EM.chest2 f32 n n)
  (#rA : EM.chest2 real n n)
  (#rB : EM.chest2 real n n)
  preserves cpu
  requires
    on gpu_loc (gA |-> sA) **
    on gpu_loc (gB |-> sB) **
    on gpu_loc (y  |-> sy) **
    pure (reveal sA %~ reveal rA /\ reveal sB %~ reveal rB)
  ensures
    on gpu_loc (gA |-> sA) **
    on gpu_loc (gB |-> sB) **
    (exists* (sy' : EM.chest2 f32 n n).
       on gpu_loc (y |-> sy') **
       pure (tril_matmul_post n (reveal rA) (reveal rB) sy'))
{
  (* Expose Seq.length / SZ.fits facts for the concrete layouts. *)
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gA);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gB);

  (* Real witness for the output buffer's initial content (the operand real
     witnesses [rA]/[rB] come from the caller via [sA %~ rA /\ sB %~ rB]). *)
  let rC : EM.chest2 real n n =
    hide (EM.to_real_matrix (reveal sy));

  assert pure (MS.comb2 #f32 `approx2` MS.comb2 #real);

  (* Launch 1: Kahan GEMM straight into the output buffer [y].  comb2 ignores
     the old [y] value, so the result is the plain real matmul
     (matmul_is_gemm SMTPat). *)
  (* Bridge Array2 ownership into tensor ownership for the new GEMM API. *)
  bridge_fwd gA;
  bridge_fwd gB;
  bridge_fwd y;

  P.mmcomb_gpu_approx (MS.comb2 #f32) (MS.comb2 #real)
    #n #n #n
    (gA) (gB) (y)
    (reveal rA) (reveal rB) (reveal rC);
  with eC'. assert on gpu_loc (y |-> eC');
  bridge_bwd gA;
  bridge_bwd gB;
  bridge_bwd y;
  assert pure (reveal eC' %~ MS.matmul (reveal rA) (reveal rB));

  (* Launch 2: in-place lower-triangular mask.  [y] now holds [s_tril eC']. *)
  Tril.tril f32 n n y;
  with sy'. assert on gpu_loc (y |-> sy');
  assert pure (reveal sy' == Tril.s_tril (reveal eC'));

  (* Discharge per-(i,j) [tril_matmul_post]. *)
  let rAB : EM.chest2 real n n =
    hide (MS.matmul (reveal rA) (reveal rB));
  Classical.forall_intro_2
    (tril_row_aux n (reveal eC') (reveal rAB) ());
  ()
}
#pop-options

let tril_matmul_f32 : tril_matmul_ty f32 = tril_matmul_f32_impl
