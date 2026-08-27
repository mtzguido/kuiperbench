module Kuiper.KB.MaxPool1D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { cmonoid_fmax_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix

(* Verified, extractable 1-D pool output-length formula (see .fst).  Provably
   equal to the pure spec [pool_out_len_1d]; the C bridge calls this instead
   of re-implementing the formula in unverified C. *)
val pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d))

(* Verification-facing wrapper type (layout-polymorphic, f32 carrier). *)
inline_for_extraction noextract
type maxpool1d_fw_ty =
  fn
  (k : szp)
  (s : szp)
  (p : szp)
  (d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l    : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) })
  (#lin  : layout2 (SZ.v bc) (SZ.v l)) {| ctlayout lin  |}
  (#lout : layout2 (SZ.v bc) (SZ.v l_out)) {| ctlayout lout |}
  (input  : array2 f32 lin  { is_global input  })
  (output : array2 f32 lout { is_global output })
  (#fIn  : perm)
  (#sx   : erased (EM.chest2 f32 (SZ.v bc) (SZ.v l)))
  (#sout : erased (EM.chest2 f32 (SZ.v bc) (SZ.v l_out)))
  requires
    cpu **
    on gpu_loc (input |-> Frac fIn sx) **
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
  ensures
    cpu **
    on gpu_loc (input |-> Frac fIn sx) **
    on gpu_loc (output |->
      windowreduce_result cmonoid_fmax_f32 sx
        (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) (SZ.v l_out))

inline_for_extraction noextract
val maxpool1d_fw_f32 : maxpool1d_fw_ty

(* Concrete-layout extractable entry (l2_row_major). *)
inline_for_extraction noextract
type maxpool1d_fw_rm_ty =
  fn
  (k : szp)
  (s : szp)
  (p : szp)
  (d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l    : szp { SZ.fits (SZ.v bc * SZ.v l) })
  (l_out : sz { SZ.v l_out == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) /\
                SZ.fits (SZ.v bc * SZ.v l_out) })
  (input  : array2 f32 (l2_row_major bc l)     { is_global input  })
  (output : array2 f32 (l2_row_major bc l_out) { is_global output })
  (#fIn  : perm)
  (#sx   : erased (EM.chest2 f32 (SZ.v bc) (SZ.v l)))
  (#sout : erased (EM.chest2 f32 (SZ.v bc) (SZ.v l_out)))
  requires
    cpu **
    on gpu_loc (input |-> Frac fIn sx) **
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
  ensures
    cpu **
    on gpu_loc (input |-> Frac fIn sx) **
    on gpu_loc (output |->
      windowreduce_result cmonoid_fmax_f32 sx
        (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d) (SZ.v l_out))

val maxpool1d_fw_rm_f32 : maxpool1d_fw_rm_ty

(* Self-allocating entry point.  Takes ONLY the raw PyTorch dims and the input
   tensor; computes [l_out], allocates the GPU output buffer, fills it, and
   returns the pair [(l_out, output_buffer)] — the buffer's ownership passes to
   the caller.  All preconditions are stated on the raw dimensions ([l + 2p]
   etc.), so the unverified bridge only performs dimension-contract checks: it
   computes nothing that feeds the kernel and allocates nothing.  Extracts to a
   C function returning a [{ uint32_t fst; float *snd; }] struct. *)
inline_for_extraction noextract
type maxpool1d_alloc_ty =
  fn
  (b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) /\
             SZ.v b * SZ.v c <= max_blocks * max_threads })
  (l : szp { SZ.fits (SZ.v b * SZ.v c * SZ.v l) })
  (k : szp)
  (s : szp)
  (p : szp)
  (d : szp)
  (input : array2 f32 (l2_row_major (b *^ c) l) { is_global input })
  (#fIn : perm)
  (#sx  : erased (EM.chest2 f32 (SZ.v (b *^ c)) (SZ.v l)))
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

val maxpool1d_alloc_f32 : maxpool1d_alloc_ty
