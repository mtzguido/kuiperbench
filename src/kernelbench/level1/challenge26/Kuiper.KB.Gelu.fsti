module Kuiper.KB.Gelu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction
let gelu_step (#t:Type0) {| floating t |} (x : t) : t =
  let half = div one (of_int 2L) in
  let rsqrt2 = div one (sqrt (of_int 2L)) in
  mul (mul half x) (add one (erf (mul x rsqrt2)))

inline_for_extraction noextract
type gelu_fw_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map gelu_step s)

val gelu_fw_f32 : gelu_fw_ty f32
val gelu_fw_f64 : gelu_fw_ty f64
