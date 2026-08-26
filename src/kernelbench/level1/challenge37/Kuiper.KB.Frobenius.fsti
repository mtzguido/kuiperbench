module Kuiper.KB.Frobenius

(* KernelBench L1 #37: Frobenius-norm normalisation, in place.
   y = x / ||x||_F where ||x||_F = sqrt(sum(x_i^2)).

   This module composes verified Kuiper primitives:
     - Kuiper.Kernel.HReduce.reduce  (sum-of-squares with a pre-map)
     - Kuiper.Kernel.Map.map_gpu     (in-place pointwise scale)

   Implementation strategy.  Use [HReduce.reduce] with [pre_map = sq_step]
   so the sum-of-squares is computed on the input directly with no
   scratch buffer.  The host then computes [inv = rsqrt sumsq] and a
   single [map_gpu] pass scales the input by [inv].

   Functional postcondition (see [Kuiper.Spec.Frobenius]): the result
   is exactly [frobenius_result inv s], where [inv] is existentially
   bound and [inv == rsqrt sumsq] for some [sumsq : t] approximating
   the real-valued sum-of-squares of the input.  This pins the result
   to be a *uniform* scaling of the input by an [inv] that
   approximates [1/||s||_F], which is genuinely Frobenius
   normalisation up to floating-point rounding.

   Edge case (all-zero input): the spec is satisfied with
   [inv = rsqrt 0]; in IEEE-754 this is [+inf] and the result is
   NaN-filled.  This matches the PyTorch reference.

   Note on input shape.  KernelBench #37 allows an input tensor of
   *arbitrary* dimension (the reference flattens it implicitly via
   [torch.norm(x)]).  We view the input as a flat [Array1] of [lena]
   elements.  This is semantically exact rather than a restriction:
   the Frobenius norm reduces over *every* element regardless of the
   logical shape, so flattening a contiguous (d0,d1,...) tensor to a
   single length-[d0*d1*...] axis computes precisely the same norm.
   The C bridge simply passes [x.numel()] as [lena]. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Frobenius
inline_for_extraction noextract
type frobenius_fw_ty (t:Type0) {| floating t, real_like t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : erased (chest1 t lena))
     requires cpu ** on gpu_loc (a |-> s)
     ensures
       cpu **
       (exists* (s' : chest1 t lena).
          on gpu_loc (a |-> s') **
          pure (exists (sumsq : t).
            chest1_to_seq s' == frobenius_result (rsqrt sumsq) (chest1_to_seq s) /\
            sumsq %~ frobenius_sumsq_r (to_real_seq (chest1_to_seq s))))

val frobenius_fw_f32 : frobenius_fw_ty f32
