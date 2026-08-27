module Kuiper.KB.MaxPool1D

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
   this instead of re-implementing the formula in unverified C.
   [s], [k], [d] are szp (>= 1) so the spec's [s = 0] branch is dead and
   the SZ subtraction [padded -^ kspan] is guarded by the [padded <^ kspan]
   test exactly as in the spec. *)
let pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d))
  =
  let kspan  : sz = SZ.((d *^ (k -^ 1sz)) +^ 1sz) in
  let padded : sz = SZ.(l +^ (2sz *^ p)) in
  if SZ.(padded <^ kspan) then 0sz
  else SZ.(((padded -^ kspan) /^ s) +^ 1sz)

inline_for_extraction noextract
fn maxpool1d_fw
  (#t : Type0) {| scalar t |}
  (m_inst : Kuiper.Monoid.Reduce.cmonoid t)
  (k s p d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l    : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) })
  (#lin  : layout2 (SZ.v bc) (SZ.v l))     {| ctlayout lin  |}
  (#lout : layout2 (SZ.v bc) (SZ.v l_out)) {| ctlayout lout |}
  (input  : array2 t lin  { is_global input  })
  (output : array2 t lout { is_global output })
  (#fIn  : perm)
  (#sx   : EM.chest2 t (SZ.v bc) (SZ.v l))
  (#sout : EM.chest2 t (SZ.v bc) (SZ.v l_out))
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
        (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) (SZ.v l_out))
{
  windowreduce m_inst k s p d bc l l_out input output
}

inline_for_extraction noextract
let maxpool1d_fw_f32 : maxpool1d_fw_ty =
  fun k s p d bc l l_out #_ #_ #_ #_ input output #fIn #sx #sout ->
    maxpool1d_fw #f32 cmonoid_fmax_f32 k s p d bc l l_out input output
      #fIn #sx #sout

let maxpool1d_fw_rm_f32 : maxpool1d_fw_rm_ty =
  fun k s p d bc l l_out input output #fIn #sx #sout ->
    maxpool1d_fw_f32 k s p d bc l l_out
      #(l2_row_major bc l)     #_
      #(l2_row_major bc l_out) #_
      input output
      #fIn #sx #sout

(* Upper bound on the pool output length: [L_out <= L + 2P].  Since the
   dilated window span [kspan = D*(K-1)+1 >= 1] and stride [S >= 1], we have
   [(padded - kspan)/S <= padded - kspan <= padded - 1 < padded].  This lets
   the self-allocating entry below discharge ALL the [l_out]-dependent
   overflow/launch-bound side conditions from bounds stated purely on the raw
   PyTorch dimensions, so the bridge never has to compute [l_out] at all. *)
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

(* Fully self-contained, self-allocating entry point.  Given ONLY the raw
   PyTorch dims and the input tensor, this computes [l_out] via the verified
   [pool_out_len_1d_sz], allocates the [(B*C, l_out)] output buffer on the GPU
   (extracts to [cudaMalloc]), fills it with the windowed max-reduction, and
   returns BOTH the output length and the freshly allocated device buffer to
   the caller.  Ownership of the returned buffer transfers to the caller (the
   bridge wraps it in a torch tensor with a [cudaFree] deleter).  The bridge
   thus performs no arithmetic and no allocation whatsoever — it only checks
   the raw-dimension contracts that appear as preconditions below. *)
inline_for_extraction noextract
fn maxpool1d_alloc
  (b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) /\
             SZ.v b * SZ.v c <= max_blocks * max_threads })
  (l : szp { SZ.fits (SZ.v b * SZ.v c * SZ.v l) })
  (k s p d : szp)
  (input : array2 f32 (l2_row_major (b *^ c) l) { is_global input })
  (#fIn : perm)
  (#sx  : EM.chest2 f32 (SZ.v (b *^ c)) (SZ.v l))
  requires
    cpu **
    on gpu_loc (input |-> Frac fIn sx) **
    pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
    pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
    pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
    pure (SZ.fits ((SZ.v l + 2 * SZ.v p) * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.fits (SZ.v b * SZ.v c * (SZ.v l + 2 * SZ.v p))) **
    pure (SZ.v b * SZ.v c * (SZ.v l + 2 * SZ.v p) <= max_blocks * max_threads)
  returns r : (lo:sz { SZ.v lo == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) }
               & array2 f32 (l2_row_major (b *^ c) lo))
  ensures
    cpu **
    on gpu_loc (input |-> Frac fIn sx) **
    on gpu_loc ((dsnd r) |->
      windowreduce_result cmonoid_fmax_f32 sx
        (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) (SZ.v (dfst r))) **
    pure (SZ.v (dfst r) ==
            pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d))
{
  let l_out = pool_out_len_1d_sz l k s p d;
  pool_out_len_1d_ub (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d);
  ML.lemma_mult_le_left (SZ.v b * SZ.v c) (SZ.v l_out) (SZ.v l + 2 * SZ.v p);
  ML.lemma_mult_le_right (SZ.v s) (SZ.v l_out) (SZ.v l + 2 * SZ.v p);
  let output = alloc0 #f32 ((b *^ c) *^ l_out) (l2_row_major (b *^ c) l_out);
  maxpool1d_fw_rm_f32 k s p d (b *^ c) l l_out input output;
  (| (l_out <: (lo:sz { SZ.v lo == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) })), output |)
}

let maxpool1d_alloc_f32 : maxpool1d_alloc_ty =
  fun b c l k s p d input #fIn #sx ->
    maxpool1d_alloc b c l k s p d input #fIn #sx
