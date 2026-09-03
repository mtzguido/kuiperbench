module Kuiper.KB.AvgPool1D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { reducer_fadd_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix

(* Verified, extractable 1-D pool output-length formula (see .fst), provably
   equal to the pure spec [pool_out_len_1d]. *)
val pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)

(* Verified, extractable reciprocal 1/k as f32 (see .fst); the average-pool
   divisor is computed inside the verification boundary. *)
val avgpool_recip_f32 (k : szp) : f32

(* Verification-facing wrapper type (layout-polymorphic, f32 carrier). *)
inline_for_extraction noextract
fn avgpool1d_fw_f32
  (k s p d : szp)
(bc : szp { SZ.v bc <= max_blocks * max_threads })
(l    : szp)
(l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
(#lin  : layout2 bc l) {| ctlayout lin  |}
(#lout : layout2 bc l_out) {| ctlayout lout |}
(input  : array2 f32 lin  { is_global input  })
(output : array2 f32 lout { is_global output })
(#fIn  : perm)
(#sx   : chest2 f32 bc l)
(#sout : chest2 f32 bc l_out)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 on gpu_loc (output |-> sout) **
 pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
 pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
ensures
 on gpu_loc (output |->
   windowreduce_result reducer_fadd_f32 sx
     k s p d l_out)


(* Concrete-layout extractable entry (l2_row_major). *)
fn avgpool1d_fw_rm_f32
  (k s p d : szp)
(bc : szp { SZ.v bc <= max_blocks * max_threads })
(l    : szp { SZ.fits (SZ.v bc * SZ.v l) })
(l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d /\
             SZ.fits (SZ.v bc * SZ.v l_out) })
(input  : array2 f32 (l2_row_major bc l)     { is_global input  })
(output : array2 f32 (l2_row_major bc l_out) { is_global output })
(#fIn  : perm)
(#sx   : chest2 f32 bc l)
(#sout : chest2 f32 bc l_out)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 on gpu_loc (output |-> sout) **
 pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
 pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
ensures
 on gpu_loc (output |->
   windowreduce_result reducer_fadd_f32 sx
     k s p d l_out)


(* Self-allocating entry point.  Takes ONLY the raw PyTorch dims and the input
   tensor; computes [l_out], allocates the GPU output buffer, fills it with the
   per-window SUM, divides every element by [K] in place with the verified
   [Kuiper.KB.ScalarMul] kernel (scaling by [inv_k = avgpool_recip_f32 k = 1/K]),
   and returns the pair [(l_out, output_buffer)] — the buffer's ownership passes
   to the caller.  All preconditions are stated on the raw dimensions
   ([l + 2p] etc.), so the unverified bridge only performs dimension-contract
   checks: it computes nothing that feeds the kernel and allocates nothing.  The
   post is exactly "windowed sum, then /K": every output accumulator equals
   [inv_k] times the corresponding [windowreduce_result] (sum) accumulator.
   Extracts to a C function returning a [{ uint32_t fst; float *snd; }] struct.

   NOTE: proving the stronger [%~ avgpool1d_post] form from [Spec.Pool1D]
   (relating this to the real-valued average over the count-include-pad window)
   is a SEPARATE deferred functional bridge; this map-of-[windowreduce_result]
   post is fully verified. *)
fn avgpool1d_alloc_f32
  (k s p d : szp)
(bc : szp { SZ.v bc <= max_blocks * max_threads })
(l : szp { SZ.fits (SZ.v bc * SZ.v l) })
(input : array2 f32 (l2_row_major bc l) { is_global input })
(#fIn : perm)
(#sx  : chest2 f32 bc l)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
 pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
 pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
 pure (SZ.fits ((SZ.v l + 2 * SZ.v p) * SZ.v s + SZ.v k * SZ.v d)) **
 pure (SZ.fits (SZ.v bc * (SZ.v l + 2 * SZ.v p))) **
 pure (SZ.v bc * (SZ.v l + 2 * SZ.v p) <= max_blocks * max_threads)
returns r : (lo:sz { SZ.v lo == pool_out_len_1d l k s p d }
            & array2 f32 (l2_row_major bc lo))
ensures
 on gpu_loc ((dsnd r) |->
   mk2 (fun (i:natlt bc) (j:natlt (dfst r)) ->
     mul (avgpool_recip_f32 k)
         (acc2 (windowreduce_result reducer_fadd_f32 sx
                    k s p d (dfst r)) i j))) **
 pure (SZ.v (dfst r) ==
         pool_out_len_1d l k s p d)

(* KernelBench-shaped entry: derives [bc = b*c] and supplies the operation's
   implicit unit dilation inside Kuiper. *)
fn avgpool1d_raw_alloc_f32
  (k s p b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) /\
             SZ.v b * SZ.v c <= max_blocks * max_threads })
  (l : szp { SZ.fits (SZ.v (b *^ c) * SZ.v l) })
  (input : array2 f32 (l2_row_major (b *^ c) l) { is_global input })
  (#fIn : perm)
  (#sx : chest2 f32 (b *^ c) l)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (SZ.fits (1 * (SZ.v k - 1) + 1)) **
    pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
    pure (1 * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
    pure (SZ.fits ((SZ.v l + 2 * SZ.v p) * SZ.v s + SZ.v k * 1)) **
    pure (SZ.fits (SZ.v (b *^ c) * (SZ.v l + 2 * SZ.v p))) **
    pure (SZ.v (b *^ c) * (SZ.v l + 2 * SZ.v p)
            <= max_blocks * max_threads)
  returns r :
    (lo : sz { SZ.v lo == pool_out_len_1d l k s p 1 }
     & array2 f32 (l2_row_major (b *^ c) lo))
  ensures
    on gpu_loc ((dsnd r) |->
      mk2 (fun (i:natlt (b *^ c)) (j:natlt (dfst r)) ->
        mul (avgpool_recip_f32 k)
          (acc2 (windowreduce_result reducer_fadd_f32 sx
            k s p 1 (dfst r)) i j))) **
    pure (SZ.v (dfst r) == pool_out_len_1d l k s p 1)
