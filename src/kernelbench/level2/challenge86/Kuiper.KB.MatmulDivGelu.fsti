module Kuiper.KB.MatmulDivGelu

(* KernelBench L2 #86: Matmul_Divide_GELU.

   PyTorch reference (nn.Linear, with bias):
     x = linear(x)        # x @ W.T + bias   -> (batch, out)
     x = x / divisor      # divide by a constant
     x = gelu(x)          # exact erf-form GELU
   Output shape (batch, out).

   PyTorch's default [torch.nn.functional.gelu] is the EXACT erf form:
     gelu(v) = 0.5 * v * (1 + erf(v / sqrt 2))

   The caller passes Wᵀ as [wt] (input × out), so the linear layer is the
   plain matmul [x @ wt] plus a per-column [bias] broadcast over rows.

   Pipeline (GPU-only, fused, three launches):
     1. GEMM      : gC := x @ wt                (Kuiper.Kernel.GEMM.Naive2.mmcomb_gpu_exact)
     2. bias-add  : y[i*out+j] := C[i,j] + bias[j]  (Kuiper.Kernel.BiasAdd.bias_add_gpu)
     3. div+gelu  : y := gelu(y / divisor)      (Kuiper.Kernel.Map.map_gpu, ONE fused map)

   The functional postcondition is EXACT (no real-number approximation): each
   output element equals the float-level composition of the three steps.  Here
   [erf]/[sqrt]/[div]/[mul]/[add] are the float operations of the [floating]
   typeclass.

   No assume / magic / admit. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Exact erf-form GELU as a total float function over the [floating]
   typeclass.  Shared between spec and impl so both reference the same
   symbol (no anonymous-lambda mismatch). *)
inline_for_extraction noextract
let gelu (#t:Type0) {| floating t |} (v : t) : t =
  let half = div one (of_int 2L) in
  let sqrt2 = sqrt (of_int 2L) in
  mul (mul v half) (add one (erf (div v sqrt2)))

(* The fused divide-then-gelu step. *)
inline_for_extraction noextract
let div_gelu (#t:Type0) {| floating t |} (divisor : t) (v : t) : t =
  gelu (div v divisor)

(* Per-element EXACT functional postcondition: the flattened output at
   [(i, j)] equals [div_gelu divisor] of the bias-augmented matmul entry. *)
let matmul_div_gelu_post
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
        div_gelu divisor (add (acc2 (MS.matmul sx swt) i j) (acc1 sbias j))

inline_for_extraction noextract
type matmul_div_gelu_ty (t:Type0) {| floating t |} =
  fn (batch input : szp)
     (out : szp {
        SZ.v batch * SZ.v out <= max_blocks * max_threads /\
        SZ.fits (SZ.v batch * SZ.v input) /\
        SZ.fits (SZ.v input * SZ.v out) /\
        SZ.fits (SZ.v batch * SZ.v out) })
     (divisor : t)
     (x    : array2 t (l2_row_major batch input) { is_global x    })
     (wt   : array2 t (l2_row_major input out)   { is_global wt   })
     (bias : array1 t (l1_forward out)                  { is_global bias })
     (y    : array1 t (l1_forward (SZ.v batch * SZ.v out))     { is_global y    })
     (#sx   : EM.chest2 t batch input)
     (#swt  : EM.chest2 t input out)
     (#sbias: chest1 t out)
     (#sy   : chest1 t (SZ.v batch * SZ.v out))
     requires
       cpu **
       on gpu_loc (x    |-> sx)   **
       on gpu_loc (wt   |-> swt)  **
       on gpu_loc (bias |-> sbias)**
       on gpu_loc (y    |-> sy)
     ensures
       cpu **
       on gpu_loc (x    |-> sx)   **
       on gpu_loc (wt   |-> swt)  **
       on gpu_loc (bias |-> sbias)**
       (exists* (sy' : chest1 t (SZ.v batch * SZ.v out)).
          on gpu_loc (y |-> sy') **
          pure (matmul_div_gelu_post divisor sx swt sbias sy'))

val matmul_div_gelu_f32 : matmul_div_gelu_ty f32
