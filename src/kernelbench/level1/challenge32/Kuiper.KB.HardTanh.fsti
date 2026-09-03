module Kuiper.KB.HardTanh

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction
let htanh_step (#t:Type0) {| floating t |} (x : t) : t =
  let hi = one in
  let lo = sub zero one in
  if gt x hi then hi else if gt lo x then lo else x

inline_for_extraction noextract
type htanh_fw_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map htanh_step s)

val htanh_fw_f32 : htanh_fw_ty f32
val htanh_fw_f64 : htanh_fw_ty f64

inline_for_extraction noextract
type htanh_alloc_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (input : array1 t (l1_forward lena) { is_global input })
     (#s : chest1 t lena)
     (#f : perm)
     norewrite
     preserves cpu ** on gpu_loc (input |-> Frac f s)
     returns output : array1 t (l1_forward lena)
     ensures on gpu_loc (output |-> mk1 (fun i -> htanh_step (acc1 s i)))

val htanh_alloc_f32 : htanh_alloc_ty f32
val htanh_alloc_f64 : htanh_alloc_ty f64
