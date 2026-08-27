module Kuiper.KB.NewGelu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
(* MinGPT NewGELU activation:
     y = 0.5 * x * (1 + tanh(c * (x + k * x^3)))
   with c = sqrt(2/pi), k = 0.044715. *)
inline_for_extraction
let newgelu_step
  (#t:Type0) {| floating t |}
  (half : t) (c : t) (k : t) (x : t) : t =
  let x3 = mul (mul x x) x in
  let inner = mul c (add x (mul k x3)) in
  mul (mul half x) (add one (tanh inner))

inline_for_extraction noextract
type newgelu_fw_ty (t:Type0) {| floating t |} =
  fn (half c k : t)
     (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map (newgelu_step half c k) s)

val newgelu_fw_f32 : newgelu_fw_ty f32
val newgelu_fw_f64 : newgelu_fw_ty f64
