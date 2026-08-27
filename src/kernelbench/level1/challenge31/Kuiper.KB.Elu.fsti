module Kuiper.KB.Elu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction
let elu_step (#t:Type0) {| floating t |} (x : t) : t =
  let alpha = one in
  if gt x zero then x else mul alpha (sub (fexp x) one)

inline_for_extraction noextract
type elu_fw_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map elu_step s)

val elu_fw_f32 : elu_fw_ty f32
val elu_fw_f64 : elu_fw_ty f64
