module Kuiper.KB.TriuMatmul

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major, c_l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module Triu = Kuiper.Kernel.Triu
module P = Kuiper.Kernel.GEMM.Naive3

(* The GEMM approximation supplies the retained entries.  Entries below the
   diagonal are exactly zero after the in-place mask. *)
#push-options "--z3rlimit 50"
let triu_row_aux
  (n : nat)
  (eC' : chest2 f32 n n)
  (rAB : chest2 real n n)
  (h_approx : squash (eC' %~ rAB))
  (i j : natlt n)
  : Lemma
      (ensures
        acc2 (Triu.s_triu eC') i j %~
          (if i <= j then acc2 rAB i j else 0.0R))
  = assert ((zero #f32) %~ 0.0R)
#pop-options

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn triu_matmul_f32_impl
  (n : szp {
     SZ.v n * SZ.v n <= max_blocks * max_threads /\
     SZ.fits (SZ.v n * SZ.v n) })
  (gA : array2 f32 (l2_row_major n n) { is_global gA })
  (gB : array2 f32 (l2_row_major n n) { is_global gB })
  (y  : array2 f32 (l2_row_major n n) { is_global y  })
  (#sA : chest2 f32 n n)
  (#sB : chest2 f32 n n)
  (#sy : chest2 f32 n n)
  (#rA : chest2 real n n)
  (#rB : chest2 real n n)
  preserves
    cpu **
    on gpu_loc (gA |-> sA) **
    on gpu_loc (gB |-> sB)
  requires
    on gpu_loc (y  |-> sy) **
    pure (reveal sA %~ reveal rA /\ reveal sB %~ reveal rB)
  ensures
    (exists* (sy' : chest2 f32 n n).
       on gpu_loc (y |-> sy') **
       pure (triu_matmul_post n (reveal rA) (reveal rB) sy'))
{
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gA);
  map_loc gpu_loc (fun () -> tensor_pts_to_ref gB);

  let rC : chest2 real n n =
    hide (EM.to_real_matrix (reveal sy));

  assert pure (MS.comb2 #f32 `approx2` MS.comb2 #real);

  P.mmcomb_gpu_approx (MS.comb2 #f32) (MS.comb2 #real)
    #n #n #n
    (gA) (gB) (y)
    (reveal rA) (reveal rB) (reveal rC);
  with eC'. assert on gpu_loc (y |-> eC');
  assert pure (reveal eC' %~ MS.matmul (reveal rA) (reveal rB));

  Triu.triu f32 n n y;
  with sy'. assert on gpu_loc (y |-> sy');
  assert pure (reveal sy' == Triu.s_triu (reveal eC'));

  let rAB : chest2 real n n =
    hide (MS.matmul (reveal rA) (reveal rB));
  Classical.forall_intro_2
    (triu_row_aux n (reveal eC') (reveal rAB) ());
  ()
}
#pop-options

let triu_matmul_f32 = triu_matmul_f32_impl
