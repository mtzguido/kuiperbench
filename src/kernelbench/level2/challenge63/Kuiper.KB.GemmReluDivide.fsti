module Kuiper.KB.GemmReluDivide

(* KernelBench L2 #63: Gemm_ReLU_Divide.

   PyTorch reference (nn.Linear, with bias):
     x = linear(x)        # x @ W.T + bias   -> (batch, out)
     x = relu(x)          # max(x, 0)
     x = x / divisor      # divide by a constant
   Output shape (batch, out).

   The caller passes Wᵀ as [wt] (input × out), so the linear layer is the
   plain matmul [x @ wt] plus a per-column [bias] broadcast over rows.

   Pipeline (GPU-only, fused, four launches):
     1. GEMM      : gC := x @ wt                (Kuiper.Kernel.GEMM.Naive2.mmcomb_gpu_exact)
     2. bias-add  : y[i*out+j] := C[i,j] + bias[j]  (Kuiper.Kernel.BiasAdd.bias_add_gpu)
     3. ReLU      : y := max(y, 0)              (Kuiper.Kernel.Map.map_gpu)
     4. divide    : y := y / divisor            (Kuiper.Kernel.Map.map_gpu)

   The functional postcondition is EXACT (no real-number approximation): each
   output element equals the float-level composition of the four steps.

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Per-element EXACT functional postcondition: the flattened output at
   [(i, j)] equals the float ReLU-then-divide of the bias-augmented matmul
   entry.  Here [div], [relu] are the float operations of the [floating]
   typeclass, [add] / [acc2] come from [scalar]. *)
let gemm_relu_div_post
  (#t:Type0) {| floating t |}
  (#batch #input #out : nat)
  (divisor : t)
  (sx  : EM.chest2 t batch input)
  (swt : EM.chest2 t input out)
  (sbias : chest1 t out)
  (sy' : chest1 t (batch * out))
  : prop
  = forall (i:natlt batch) (j:natlt out).
      acc1 sy' (i * out + j) ==
        div (relu (add (acc2 (MS.matmul sx swt) i j) (acc1 sbias j))) divisor

fn gemm_relu_divide_f32
  (batch input : szp)
  (out : szp {
     SZ.v batch * SZ.v out <= max_blocks * max_threads /\
     SZ.fits (SZ.v batch * SZ.v input) /\
     SZ.fits (SZ.v input * SZ.v out) /\
     SZ.fits (SZ.v batch * SZ.v out) })
  (divisor : f32)
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
       pure (gemm_relu_div_post divisor sx swt sbias sy'))
