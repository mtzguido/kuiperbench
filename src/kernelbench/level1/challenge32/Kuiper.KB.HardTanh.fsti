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
     (#s : erased (chest1 t lena))
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map htanh_step s)

val htanh_fw_f32 : htanh_fw_ty f32
val htanh_fw_f64 : htanh_fw_ty f64
