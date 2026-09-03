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

inline_for_extraction noextract
type gelu_alloc_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (input : array1 t (l1_forward lena) { is_global input })
     (#s : chest1 t lena)
     (#f : perm)
     norewrite
     preserves cpu ** on gpu_loc (input |-> Frac f s)
     returns output : array1 t (l1_forward lena)
     ensures on gpu_loc (output |-> mk1 (fun i -> gelu_step (acc1 s i)))

val gelu_alloc_f32 : gelu_alloc_ty f32
val gelu_alloc_f64 : gelu_alloc_ty f64
