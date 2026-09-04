module Kuiper.KB.AvgPool2D

(* KernelBench L1 #45: AvgPool2D.
 *
 * 2-D average pooling (with PyTorch's default count_include_pad=True
 * and stride defaulting to kernel_size) reduces to two passes of the
 * verified [Kuiper.Kernel.WindowReduce1D] primitive instantiated with
 * [reducer_fadd_f32] (rid = 0.0f, rop = +), followed by verified in-place
 * scaling by the f32 reciprocal of the corresponding kernel extent.
 *
 *   pass 1: per-row sum over W (B*C*H rows of length W); then scale /kW
 *   pass 2: per-row sum over H (B*C*W_out rows of length H); then scale /kH
 *
 * The public entry below verifies both complete separable passes, the
 * zero-copy layout recast between them, both scales, and concatenation of the
 * two ABI-sized input halves.  Its post is the exact implementation-order f32
 * result; no floating-point associativity is assumed.
 *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout { from_seq, to_seq, is_full }
open Kuiper.Tensor.Layout.Alg { l2_row_major, l1_forward }
open Kuiper.Monoid.Reduce.F32 { reducer_fadd_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix

open Kuiper.Tensor.Layout.BCMPages { l2_bcm_pages }

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
val avgpool_recip_f32 (k : szp)
  : r:f32 { r %~ (1.0R /. FStar.Real.of_int (SZ.v k)) }

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
   windowreduce_result reducer_fadd_f32 sx
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
   windowreduce_result reducer_fadd_f32 sx
     k s p d l_out)


(* Exact result of scaling every physical slot of a full layout. *)
unfold
let avgpool2d_scale_layout_result
  (#rows #cols : nat)
  (layout : layout2 rows cols { is_full layout })
  (c : f32) (sx : chest2 f32 rows cols)
  : chest2 f32 rows cols
  = let flat = l1_forward (rows * cols) in
    from_seq layout
      (to_seq flat (chest_map (mul c) (from_seq flat (to_seq layout sx))))

[@@"opaque_to_smt"]
let avgpool2d_axis_layout_result
  (#rows #l #l_out : nat)
  (layout : layout2 rows l_out { is_full layout })
  (k : szp) (sx : chest2 f32 rows l) (s p d : nat)
  : chest2 f32 rows l_out
  = avgpool2d_scale_layout_result layout (avgpool_recip_f32 k)
      (windowreduce_result reducer_fadd_f32 sx k s p d l_out)

[@@"opaque_to_smt"]
let avgpool2d_mid_w_view
  (bc h w : nat)
  (sx : chest2 f32 (bc * h) w)
  : chest2 f32 (bc * 186) h
  = from_seq (l2_bcm_pages bc 186 h)
      (to_seq (l2_row_major (bc * h) 186)
        (avgpool2d_axis_layout_result
          (l2_row_major (bc * h) 186) 11sz sx 11 0 1))

[@@"opaque_to_smt"]
let avgpool2d_half_result
  (bc h w : nat)
  (sx : chest2 f32 (bc * h) w)
  : chest2 f32 (bc * 186) 186
  = avgpool2d_axis_layout_result (l2_bcm_pages bc 186 186) 11sz
      (avgpool2d_mid_w_view bc h w sx) 11 0 1

(* Logical concatenation, independent of the two physical allocations used
   for the SizeT representation boundary. *)
unfold
let avgpool2d_concat_result
  (#n:nat) (x0 x1 : chest1 f32 n) : chest1 f32 (n + n)
  = mk1 (fun (i:natlt (n + n)) ->
      if i < n then acc1 x0 i else acc1 x1 (i - n))

(* Private compile-time dimensions keep the public ABI pointer-only while
   giving the internal SizeT-indexed layouts definitionally identical names. *)
inline_for_extraction noextract
val avgpool2d_canonical_bc : x:szp { SZ.v x == 512 }

inline_for_extraction noextract
val avgpool2d_canonical_h : x:szp { SZ.v x == 2048 }

inline_for_extraction noextract
val avgpool2d_canonical_w : x:szp { SZ.v x == 2048 }

inline_for_extraction noextract
val avgpool2d_canonical_half_n : x:szp { SZ.v x == 512 * 186 * 186 }

(* Complete canonical KernelBench #45 computation.  The 2^32-element input
   cannot be represented by one strict-SizeT array, so the ABI supplies two
   disjoint 2^31-element halves.  This is only a representation boundary:
   every pool pass, recast, allocation, scale, and the final logical
   concatenation occurs in this one verified entry. *)
fn avgpool2d_full_alloc_f32
  (input0 : array2 f32
    (l2_row_major (avgpool2d_canonical_bc * avgpool2d_canonical_h)
      avgpool2d_canonical_w) { is_global input0 })
  (input1 : array2 f32
    (l2_row_major (avgpool2d_canonical_bc * avgpool2d_canonical_h)
      avgpool2d_canonical_w) { is_global input1 })
  (#f0 #f1 : perm)
  (#sx0 #sx1 : chest2 f32
    (avgpool2d_canonical_bc * avgpool2d_canonical_h)
    avgpool2d_canonical_w)
preserves
  cpu **
  on gpu_loc (input0 |-> Frac f0 sx0) **
  on gpu_loc (input1 |-> Frac f1 sx1)
returns out : array1 f32
  (l1_forward
    (SZ.v avgpool2d_canonical_half_n + SZ.v avgpool2d_canonical_half_n))
ensures
  on gpu_loc (out |->
    avgpool2d_concat_result
      (from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
        (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
          (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
            (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx0)))
      (from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
        (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
          (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
            (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx1))))
