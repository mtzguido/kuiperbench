module Kuiper.KB.MaxPool3D

(* KernelBench L1 #43: MaxPool3D.
 *
 * 3-D max pooling reduces to three passes of the verified
 * [Kuiper.Kernel.WindowReduce1D] primitive instantiated with
 * [cmonoid_fmax_f32] (rid = -inf, rop = fmaxf):
 *
 *   pass 1: per-row max over W axis (B*C*D*H rows of length W)
 *   pass 2: per-row max over H axis (B*C*D*W_out rows of length H,
 *           after a host-side permute)
 *   pass 3: per-row max over D axis (B*C*W_out*H_out rows of length D,
 *           after a second host-side permute)
 *
 * Composing the three passes gives the 3-D max because [fmaxf] is
 * associative and commutative.  Padding contributes -inf in each pass,
 * so the composition's padding semantics match PyTorch nn.MaxPool3d.
 *
 * This .fsti exposes a single extracted entry [maxpool3d_axis_fw_rm_f32]
 * which is the verified per-axis pass; the C++ bridge orchestrates the
 * three calls and the inter-pass permute (PyTorch's transpose+contiguous).
 * The "axis" name reflects that the kernel operates on the *inner*
 * (last) axis of a 2-D row-major view; the bridge supplies whichever
 * axis is currently inner.
 *)

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
   equal to the pure spec [pool_out_len_1d]; the C bridge calls this (per axis)
   instead of re-implementing the formula in unverified C. *)
val pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)

(* Verification-facing wrapper type (layout-polymorphic, f32 carrier). *)
inline_for_extraction noextract
fn maxpool3d_axis_fw_f32
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
   windowreduce_result cmonoid_fmax_f32 sx
     k s p d l_out)


(* Concrete-layout extractable entry (l2_row_major). *)
fn maxpool3d_axis_fw_rm_f32
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
   windowreduce_result cmonoid_fmax_f32 sx
     k s p d l_out)


(* Self-allocating per-axis pass.  Takes the flattened row count [bc] and the
   inner axis length [l] (NOT a pre-computed [l_out], NOT a caller buffer);
   computes [l_out], allocates the [(bc, l_out)] GPU output buffer, fills it,
   and returns the pair [(l_out, output_buffer)] — ownership passes to the
   caller.  All preconditions are on the raw axis dims, so the bridge only
   performs dimension-contract checks and the (unavoidable) inter-pass permute
   copies; it computes no output length, allocates no pool buffer, and runs no
   launch loop.  Extracts to a C function returning a
   [{ uint32_t fst; float *snd; }] struct. *)
fn maxpool3d_axis_alloc_f32
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
   windowreduce_result cmonoid_fmax_f32 sx
     k s p d (dfst r)) **
 pure (SZ.v (dfst r) ==
         pool_out_len_1d l k s p d)
