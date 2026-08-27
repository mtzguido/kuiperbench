module Kuiper.KB.AvgPool2D

(* KernelBench L1 #45: AvgPool2D.
 *
 * 2-D average pooling (with PyTorch's default count_include_pad=True
 * and stride defaulting to kernel_size) reduces to two passes of the
 * verified [Kuiper.Kernel.WindowReduce1D] primitive instantiated with
 * [cmonoid_fadd_f32] (rid = 0.0f, rop = +) plus a per-pass
 * (unverified) scale by 1/k.
 *
 *   pass 1: per-row sum over W (B*C*H rows of length W); then scale /kW
 *   pass 2: per-row sum over H (B*C*W_out rows of length H); then scale /kH
 *
 * Composition: real-valued sum is associative+commutative so the
 * 2-D avg = sum/(kH*kW); applying /kW after pass 1 and /kH after
 * pass 2 yields /(kH*kW).  The verified primitive's post is the
 * exact-equality fold over [cmonoid_fadd_f32]; the cmonoid's
 * associativity/commutativity rests on the same f32 axioms as the
 * 1-D variant (#44), and approximation to the real sum follows the
 * existing %~ pattern.
 *
 * Same pattern as #44 AvgPool1D: a verified per-axis sum kernel +
 * an unverified scale post-pass.  The C++ bridge orchestrates the
 * two passes, the inter-pass permute, and the two scale post-passes.
 *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { cmonoid_fadd_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix

(* Verified, extractable 1-D pool output-length formula (see .fst), provably
   equal to the pure spec [pool_out_len_1d].  [p] is [sz] (>= 0) since 2-D
   avg-pool uses [P = 0] by default. *)
val pool_out_len_1d_sz
  (l k s : szp) (p : sz) (d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)

(* Verified, extractable reciprocal 1/k as f32 (see .fst). *)
val avgpool_recip_f32 (k : szp) : f32

(* Verification-facing wrapper type (layout-polymorphic, f32 carrier). *)
inline_for_extraction noextract
fn avgpool2d_axis_fw_f32
  (k s : szp)
(p : sz)
(d : szp)
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
   windowreduce_result cmonoid_fadd_f32 sx
     k s p d l_out)


(* Concrete-layout extractable entry (l2_row_major). *)
fn avgpool2d_axis_fw_rm_f32
  (k s : szp)
(p : sz)
(d : szp)
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
   windowreduce_result cmonoid_fadd_f32 sx
     k s p d l_out)


(* Self-allocating per-axis entry point.  Takes ONLY the raw per-axis PyTorch
   dims [(k, s, p, d)], the leading product [bc] (the non-reduced dims folded
   into rows for this pass), the reduced axis length [l], and the input tensor.
   It computes [l_out] via the verified [pool_out_len_1d_sz], allocates the
   [(bc, l_out)] GPU output buffer, fills it with the per-window SUM
   (cmonoid_fadd_f32, rid = 0, padding -> 0), divides every element by [K] in
   place via the verified [Kuiper.KB.ScalarMul] kernel (scaling by
   [inv_k = avgpool_recip_f32 k = 1/K]), and returns the pair
   [(l_out, output_buffer)] — ownership passes to the caller.  The post is
   exactly "windowed sum, then /K": every output accumulator equals [inv_k]
   times the corresponding [windowreduce_result] (sum) accumulator.  Applying
   this per-pass /K across the two (2-D) separable axis passes yields the
   PyTorch divisor K*K (count_include_pad = True).

   All preconditions are stated on the raw per-axis dims and on [bc * l_out]
   (the verified [pool_out_len_1d] of the axis), so the unverified bridge only
   performs dimension-contract checks and computes [l_out] via the *verified*
   [pool_out_len_1d_sz]; it allocates nothing and launches no second kernel.

   Extracts to a C function returning a [{ uint32_t fst; float *snd; }] struct. *)
fn avgpool2d_axis_alloc_f32
  (k s : szp)
(p : sz)
(d : szp)
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
 pure (SZ.fits (pool_out_len_1d l k s p d
                  * SZ.v s + SZ.v k * SZ.v d)) **
 pure (SZ.fits (SZ.v bc *
         pool_out_len_1d l k s p d)) **
 pure (SZ.v bc *
         pool_out_len_1d l k s p d
       <= max_blocks * max_threads)
returns r : (lo:sz { SZ.v lo == pool_out_len_1d l k s p d }
            & array2 f32 (l2_row_major bc lo))
ensures
 on gpu_loc ((dsnd r) |->
   mk2 (fun (i:natlt bc) (j:natlt (dfst r)) ->
     mul (avgpool_recip_f32 k)
         (acc2 (windowreduce_result cmonoid_fadd_f32 sx
                    k s p d (dfst r)) i j))) **
 pure (SZ.v (dfst r) ==
         pool_out_len_1d l k s p d)
