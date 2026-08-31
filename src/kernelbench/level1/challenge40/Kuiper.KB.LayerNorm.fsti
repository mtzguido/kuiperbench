module Kuiper.KB.LayerNorm

(* KernelBench L1 #40: row-wise LayerNorm, in place.

   Treats the input as a flat (b * n) array, viewed as [b] rows of [n].
   For each row r, computes
       mean_r = (1/n) * Σ_j  x[r,j]
       var_r  = (1/n) * Σ_j  x[r,j]^2  - mean_r^2
       inv_r  = 1 / sqrt(var_r + eps)
       y[r,j] = ((x[r,j] - mean_r) * inv_r) * γ[j] + β[j]

   Composes verified Kuiper primitives:
     - Kuiper.Scalars.square          (pointwise square)
     - Kuiper.Kernel.HReduce.reduce  (sum)
     - Kuiper.Kernel.Map.map_gpu      (apply affine_step (inv, -mean*inv) per row)
     - Kuiper.Kernel.Map.map_gpu2    (γ then β broadcast: scratch *= γ; scratch += β)

   Per-row uses the device-to-device offset memcpy primitive to copy a
   row in/out of a fixed-size scratch buffer.  γ and β are held with
   fractional permission throughout (read-only across the row loop).
   The direct real proof uses the temporary [rsqrt_approx] compatibility
   assumption documented in the repository patch.  The existing
   [map_gpu2] sendability debt is documented in the module's skeptic. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.LayerNorm
module SZ = Kuiper.SizeT

(* The per-row reciprocal 1/n, built from the runtime [n] inside the
   verification boundary (extracts to 1.0f / (float)(int64_t)(uint64_t)n)
   so no unverified floating-point arithmetic happens in the C bridge. *)
inline_for_extraction noextract
let ln_inv_n (#t:Type0) {| floating t |} (n : SZ.t) : t =
  div one (of_int (FStar.Int.Cast.uint64_to_int64
                     (FStar.SizeT.sizet_to_uint64 n)))

fn layernorm_fw_f32
  (b : szp)
  (n : szp { n <= max_blocks * max_threads /\
             SZ.fits (b * n) /\
             b * n <= max_blocks * max_threads })
  (eps : f32)
  (x     : array1 f32 (l1_forward (b * n)) { is_global x     })
  (gamma : array1 f32 (l1_forward n)        { is_global gamma })
  (beta  : array1 f32 (l1_forward n)        { is_global beta  })
  (#fg #fb : perm)
  (#sx : chest1 f32 (b * n))
  (#sg #sb : chest1 f32 n)
  preserves
    cpu **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sb)
  requires
    on gpu_loc (x |-> sx) **
    pure (layernorm_domain b n eps (chest1_to_seq sx))
  ensures
    (exists* (sx' : chest1 f32 (b * n)).
       on gpu_loc (x |-> sx') **
       pure (layernorm_post b n eps
               (chest1_to_seq sg) (chest1_to_seq sb)
               (chest1_to_seq sx) (chest1_to_seq sx')))
