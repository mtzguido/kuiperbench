module Kuiper.KB.MaxPool3D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { cmonoid_fmax_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce, windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module ML = FStar.Math.Lemmas

(* Verified, extractable computation of the PyTorch 1-D pool output length,
   provably equal to the pure spec [pool_out_len_1d].  The C bridge calls
   this (per axis) instead of re-implementing the formula in unverified C.
   [s], [k], [d] are szp (>= 1) so the spec's [s = 0] branch is dead and
   the SZ subtraction [padded -^ kspan] is guarded by the [padded <^ kspan]
   test exactly as in the spec. *)
let pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)
  =
  let kspan  : sz = SZ.((d *^ (k -^ 1sz)) +^ 1sz) in
  let padded : sz = SZ.(l +^ (2sz *^ p)) in
  if SZ.(padded <^ kspan) then 0sz
  else SZ.(((padded -^ kspan) /^ s) +^ 1sz)

inline_for_extraction noextract
fn maxpool3d_axis_fw
  (#t : Type0) {| scalar t |}
  (m_inst : Kuiper.Monoid.Reduce.cmonoid t)
  (k s p d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l    : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
  (#lin  : layout2 bc l)     {| ctlayout lin  |}
  (#lout : layout2 bc l_out) {| ctlayout lout |}
  (input  : array2 t lin  { is_global input  })
  (output : array2 t lout { is_global output })
  (#fIn  : perm)
  (#sx   : EM.chest2 t bc l)
  (#sout : EM.chest2 t bc l_out)
  requires
    cpu **
    on gpu_loc (input  |-> Frac fIn sx) **
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
  ensures
    cpu **
    on gpu_loc (input  |-> Frac fIn sx) **
    on gpu_loc (output |->
      windowreduce_result m_inst sx
        k s p d l_out)
{
  windowreduce m_inst k s p d bc l l_out input output
}

inline_for_extraction noextract
let maxpool3d_axis_fw_f32 : maxpool3d_axis_fw_ty =
  fun k s p d bc l l_out #_ #_ #_ #_ input output #fIn #sx #sout ->
    maxpool3d_axis_fw #f32 cmonoid_fmax_f32 k s p d bc l l_out input output
      #fIn #sx #sout

let maxpool3d_axis_fw_rm_f32 : maxpool3d_axis_fw_rm_ty =
  fun k s p d bc l l_out input output #fIn #sx #sout ->
    maxpool3d_axis_fw_f32 k s p d bc l l_out
      #(l2_row_major bc l)     #_
      #(l2_row_major bc l_out) #_
      input output
      #fIn #sx #sout

(* Upper bound on the pool output length: [L_out <= L + 2P].  [kspan >= 1] and
   [S >= 1] give [(padded - kspan)/S <= padded - kspan <= padded - 1 < padded].
   This lets the self-allocating axis entry below discharge ALL the
   [l_out]-dependent overflow / launch-bound side conditions from bounds stated
   purely on the raw axis dims [(bc, l)], so the bridge never computes
   [l_out]. *)
let pool_out_len_1d_ub (l k s p d : nat)
  : Lemma (requires k >= 1 /\ s >= 1 /\ d >= 1)
          (ensures pool_out_len_1d l k s p d <= l + 2 * p)
  = let kspan = d * (k - 1) + 1 in
    let padded = l + 2 * p in
    if padded < kspan || s = 0 then ()
    else begin
      ML.lemma_div_mod (padded - kspan) s;
      assert ((padded - kspan) / s <= padded - kspan)
    end

(* Self-allocating per-axis pass.  Given the flattened row count [bc], the inner
   axis length [l], and the pool params, this computes [l_out] via the verified
   [pool_out_len_1d_sz], allocates the [(bc, l_out)] GPU output buffer (extracts
   to cudaMalloc), runs the fmax windowreduce, and returns BOTH [l_out] and the
   freshly allocated buffer.  Ownership of the buffer transfers to the caller
   (the bridge wraps it in a torch tensor with a cudaFree deleter, permutes, and
   feeds it to the next axis pass).  The bridge thus computes no output length,
   allocates no pool buffer, and runs no launch loop; it only supplies the
   flattened row count and inner length and checks raw-dimension contracts. *)
inline_for_extraction noextract
fn maxpool3d_axis_alloc
  (k s p d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l : szp { SZ.fits (SZ.v bc * SZ.v l) })
  (input : array2 f32 (l2_row_major bc l) { is_global input })
  (#fIn : perm)
  (#sx  : EM.chest2 f32 bc l)
  requires
    cpu **
    on gpu_loc (input |-> Frac fIn sx) **
    pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
    pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
    pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
    pure (SZ.fits ((SZ.v l + 2 * SZ.v p) * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.fits (SZ.v bc * (SZ.v l + 2 * SZ.v p))) **
    pure (SZ.v bc * (SZ.v l + 2 * SZ.v p) <= max_blocks * max_threads)
  returns r : (lo:sz { SZ.v lo == pool_out_len_1d l k s p d }
               & array2 f32 (l2_row_major bc lo))
  ensures
    cpu **
    on gpu_loc (input |-> Frac fIn sx) **
    on gpu_loc ((dsnd r) |->
      windowreduce_result cmonoid_fmax_f32 sx
        k s p d (dfst r)) **
    pure (SZ.v (dfst r) ==
            pool_out_len_1d l k s p d)
{
  let l_out = pool_out_len_1d_sz l k s p d;
  pool_out_len_1d_ub l k s p d;
  ML.lemma_mult_le_left bc l_out (SZ.v l + 2 * SZ.v p);
  ML.lemma_mult_le_right s l_out (SZ.v l + 2 * SZ.v p);
  let output = alloc0 #f32 (bc *^ l_out) (l2_row_major bc l_out);
  maxpool3d_axis_fw_rm_f32 k s p d bc l l_out input output;
  (| (l_out <: (lo:sz { SZ.v lo == pool_out_len_1d l k s p d })), output |)
}

let maxpool3d_axis_alloc_f32 : maxpool3d_axis_alloc_ty =
  fun k s p d bc l input #fIn #sx ->
    maxpool3d_axis_alloc k s p d bc l input #fIn #sx
