module Kuiper.KB.MaxPool1D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { reducer_fmax_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix

noeq type maxpool1d_alloc_result
  (b c l k s p d : szp) = {
  l_out : lo:sz { SZ.v lo == pool_out_len_1d l k s p d };
  output : array2 f32 (l2_row_major (b * c) l_out);
}

(* Verified, extractable 1-D pool output-length formula (see .fst).  Provably
   equal to the pure spec [pool_out_len_1d]; the C bridge calls this instead
   of re-implementing the formula in unverified C. *)
val pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)

(* Verification-facing wrapper type (layout-polymorphic, f32 carrier). *)
inline_for_extraction noextract
fn maxpool1d_fw_f32
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
   windowreduce_result reducer_fmax_f32 sx
     k s p d l_out)


(* Concrete-layout extractable entry (l2_row_major). *)
fn maxpool1d_fw_rm_f32
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
   windowreduce_result reducer_fmax_f32 sx
     k s p d l_out)


(* Self-allocating entry point.  Takes ONLY the raw PyTorch dims and the input
   tensor; computes [l_out], allocates the GPU output buffer, fills it, and
   returns both in a named result record — the buffer's ownership passes to
   the caller.  All preconditions are stated on the raw dimensions ([l + 2p]
   etc.), so the unverified bridge only performs dimension-contract checks: it
   computes nothing that feeds the kernel and allocates nothing. *)
fn maxpool1d_alloc_f32
  (b : szp)
(c : szp { SZ.fits (SZ.v b * SZ.v c) /\
          SZ.v b * SZ.v c <= max_blocks * max_threads })
(l : szp { SZ.fits (SZ.v b * SZ.v c * SZ.v l) })
(k s p d : szp)
(input : array2 f32 (l2_row_major (b * c) l) { is_global input })
(#fIn : perm)
(#sx  : chest2 f32 (b * c) l)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
 pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
 pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
 pure (SZ.fits ((SZ.v l + 2 * SZ.v p) * SZ.v s + SZ.v k * SZ.v d)) **
 pure (SZ.fits (SZ.v b * SZ.v c * (SZ.v l + 2 * SZ.v p))) **
 pure (SZ.v b * SZ.v c * (SZ.v l + 2 * SZ.v p) <= max_blocks * max_threads)
returns r : maxpool1d_alloc_result b c l k s p d
ensures
 on gpu_loc (r.output |->
   windowreduce_result reducer_fmax_f32 sx
     k s p d r.l_out) **
 pure (SZ.v r.l_out ==
         pool_out_len_1d l k s p d)
