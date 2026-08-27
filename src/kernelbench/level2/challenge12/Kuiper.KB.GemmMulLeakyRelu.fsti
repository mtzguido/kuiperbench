module Kuiper.KB.GemmMulLeakyRelu

(* KernelBench L2 #12: Gemm_Multiply_LeakyReLU.

   PyTorch reference (nn.Linear, with bias):
     x = gemm(x)          # x @ W.T + bias   -> (batch, out)
     x = x * multiplier   # multiply by a constant
     x = leaky_relu(x)    # x if x>=0 else negative_slope*x
   Output shape (batch, out).

   The caller passes Wᵀ as [wt] (input × out), so the linear layer is the
   plain matmul [x @ wt] plus a per-column [bias] broadcast over rows.

   Pipeline (GPU-only, fused, three launches):
     1. GEMM      : gC := x @ wt                (Kuiper.Kernel.GEMM.Naive2.mmcomb_gpu_exact)
     2. bias-add  : y[i*out+j] := C[i,j] + bias[j]  (Kuiper.Kernel.BiasAdd.bias_add_gpu)
     3. mul+lrelu : y := leaky_relu (y * multiplier)  (one fused Kuiper.Kernel.Map.map_gpu)

   The functional postcondition is EXACT (no real-number approximation): each
   output element equals the float-level composition of the steps, written via
   the SAME [mul_lrelu] / [leaky_relu] closures used by the implementation.

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* LeakyReLU as a float-typeclass closure: [v] if [v >= 0] else [slope * v].
   An EXACT float branch is fine here because the spec is float-level [==]. *)
inline_for_extraction noextract
let leaky_relu (#t:Type0) {| floating t |} (slope v : t) : t =
  if gte v zero then v else mul v slope

(* Fused multiply-by-constant then LeakyReLU. *)
inline_for_extraction noextract
let mul_lrelu (#t:Type0) {| floating t |} (mult slope : t) (v : t) : t =
  leaky_relu slope (mul v mult)

(* Per-element EXACT functional postcondition: the flattened output at [(i, j)]
   equals the fused multiply-then-LeakyReLU of the bias-augmented matmul entry. *)
let gemm_mul_lrelu_post
  (#t:Type0) {| floating t |}
  (#batch #input #out : nat)
  (mult slope : t)
  (sx  : EM.chest2 t batch input)
  (swt : EM.chest2 t input out)
  (sbias : chest1 t out)
  (sy' : chest1 t (batch * out))
  : prop
  = forall (i:natlt batch) (j:natlt out).
      acc1 sy' (i * out + j) ==
        mul_lrelu mult slope (add (acc2 (MS.matmul sx swt) i j) (acc1 sbias j))

fn gemm_mul_leaky_relu_f32
  (batch input : szp)
  (out : szp {
     SZ.v batch * SZ.v out <= max_blocks * max_threads /\
     SZ.fits (SZ.v batch * SZ.v input) /\
     SZ.fits (SZ.v input * SZ.v out) /\
     SZ.fits (SZ.v batch * SZ.v out) })
  (mult slope : f32)
  (x    : array2 f32 (l2_row_major batch input) { is_global x    })
  (wt   : array2 f32 (l2_row_major input out)   { is_global wt   })
  (bias : array1 f32 (l1_forward out)                  { is_global bias })
  (y    : array1 f32 (l1_forward (SZ.v batch * SZ.v out))     { is_global y    })
  (#sx   : EM.chest2 f32 batch input)
  (#swt  : EM.chest2 f32 input out)
  (#sbias: chest1 f32 out)
  (#sy   : chest1 f32 (SZ.v batch * SZ.v out))
  preserves
    cpu **
    on gpu_loc (x    |-> sx) **
    on gpu_loc (wt   |-> swt) **
    on gpu_loc (bias |-> sbias)
  requires
    on gpu_loc (y    |-> sy)
  ensures
    (exists* (sy' : chest1 f32 (SZ.v batch * SZ.v out)).
       on gpu_loc (y |-> sy') **
       pure (gemm_mul_lrelu_post mult slope sx swt sbias sy'))
