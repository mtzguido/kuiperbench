module Kuiper.KB.MaxPool2D

(* KernelBench L1 #42: MaxPool2D.
 *
 * 2-D max pooling reduces to two passes of the verified
 * [Kuiper.Kernel.WindowReduce1D] primitive instantiated with
 * [reducer_fmax_f32] (rid = -inf, rop = fmaxf):
 *
 *   pass 1: per-row max over W axis (B*C*H rows of length W)
 *   pass 2: per-row max over H axis (B*C*W_out rows of length H,
 *           after a host-side (B,C,H,W_out) -> (B,C,W_out,H) permute)
 *
 * Each pass has an exact implementation-order left-fold specification.
 * The C++ bridge composes the passes and the KernelBench oracle checks the
 * resulting 2-D operation; this interface assumes no [fmax] algebra.
 *
 * This .fsti exposes a single extracted entry [maxpool2d_axis_fw_rm_f32]
 * which is the verified per-axis pass; the C++ bridge orchestrates the
 * two calls and the inter-pass permute (PyTorch's transpose+contiguous).
 * The "axis" name reflects that the kernel operates on the *inner*
 * (last) axis of a 2-D row-major view; the bridge supplies whichever
 * axis is currently inner.
 *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout { from_seq, to_seq }
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { reducer_fmax_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
open Kuiper.Tensor.Layout.BCMPages { l2_bcm_pages }
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

(* The exact ghost value presented to the height pass after the width-pass
   result is reinterpreted through the BCM-pages layout.  The recast moves no
   data: [to_seq] exposes the width-pass buffer in physical order and
   [from_seq] views those same elements as (bc * w_out, h). *)
unfold
let maxpool2d_mid_view
  (bc h w : nat)
  (kw sw : pos) (pw : nat) (dw : pos)
  (w_out : pos)
  (sx : chest2 f32 (bc * h) w)
  : chest2 f32 (bc * w_out) h
  = from_seq (l2_bcm_pages bc w_out h)
      (to_seq (l2_row_major (bc * h) w_out)
        (windowreduce_result reducer_fmax_f32 sx
          kw sw pw dw w_out))

(* Verification-facing wrapper type (layout-polymorphic, f32 carrier). *)
inline_for_extraction noextract
fn maxpool2d_axis_fw_f32
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
fn maxpool2d_axis_fw_rm_f32
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


(* Self-allocating per-axis pass.  Takes the flattened row count [bc] and the
   inner axis length [l] (NOT a pre-computed [l_out], NOT a caller buffer);
   computes [l_out], allocates the [(bc, l_out)] GPU output buffer, fills it,
   and returns the pair [(l_out, output_buffer)] — ownership passes to the
   caller.  All preconditions are on the raw axis dims, so the bridge only
   performs dimension-contract checks and the (unavoidable) inter-pass permute
   copies; it computes no output length and allocates no pool buffer.  Extracts
   to a C function returning a [{ uint32_t fst; float *snd; }] struct. *)
fn maxpool2d_axis_alloc_f32
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
   windowreduce_result reducer_fmax_f32 sx
     k s p d (dfst r)) **
 pure (SZ.v (dfst r) ==
         pool_out_len_1d l k s p d) **
 pure (is_global (dsnd r)) **
 pure (is_full_array (core (dsnd r)))


(* ── Single verified, transpose-free 2-D max-pool entry ──────────────

   [maxpool2d_full_alloc_f32] folds the WHOLE separable 2-D max pool into
   one verified F*/Pulse call, eliminating the unverified PyTorch
   [.permute().contiguous()] that used to sit between the two
   [windowreduce] passes in the C++ bridge.

   Input is a row-major (B*C*H, W) view ([bc = B*C], inner = W).  The
   returned (W_out, H_out, buffer) triple's buffer is physically the
   row-major (B*C, H_out, W_out) result.  Its post composes both passes:
   [maxpool2d_mid_view] is exactly the width-pass result viewed as
   (B*C*W_out, H), and the returned buffer is its height-pass result. *)
fn maxpool2d_full_alloc_f32
  (kh kw sh sw ph pw dh dw bc h w : szp)
(#_ : squash (SZ.fits (SZ.v bc * SZ.v h)))
(input : array2 f32 (l2_row_major (bc * h) w) { is_global input })
(#fIn : perm)
(#sx  : chest2 f32 (bc * h) w)
preserves
 cpu **
 on gpu_loc (input |-> Frac fIn sx)
requires
 pure (SZ.fits (SZ.v dw * (SZ.v kw - 1) + 1)) **
 pure (SZ.fits (SZ.v dh * (SZ.v kh - 1) + 1)) **
 pure (SZ.fits (SZ.v w + 2 * SZ.v pw)) **
 pure (SZ.fits (SZ.v h + 2 * SZ.v ph)) **
 pure (SZ.v dw * (SZ.v kw - 1) + 1 <= SZ.v w + 2 * SZ.v pw) **
 pure (SZ.v dh * (SZ.v kh - 1) + 1 <= SZ.v h + 2 * SZ.v ph) **
 pure (SZ.fits ((SZ.v w + 2 * SZ.v pw) * SZ.v sw + SZ.v kw * SZ.v dw)) **
 pure (SZ.fits ((SZ.v h + 2 * SZ.v ph) * SZ.v sh + SZ.v kh * SZ.v dh)) **
 pure (SZ.fits (SZ.v bc * (SZ.v h + 2 * SZ.v ph) * (SZ.v w + 2 * SZ.v pw))) **
 pure (SZ.v bc * (SZ.v h + 2 * SZ.v ph) * (SZ.v w + 2 * SZ.v pw)
         <= max_blocks * max_threads)
returns r : (wo : sz { SZ.v wo == pool_out_len_1d w kw sw pw dw
                      /\ SZ.v wo > 0 }
            & (ho : sz { SZ.v ho == pool_out_len_1d h kh sh ph dh
                         /\ SZ.v ho > 0 }
               & array2 f32 (l2_bcm_pages bc wo ho)))
ensures
 on gpu_loc ((dsnd (dsnd r)) |->
   windowreduce_result reducer_fmax_f32
     (maxpool2d_mid_view bc h w kw sw pw dw (dfst r) sx)
     kh sh ph dh (dfst (dsnd r)))
