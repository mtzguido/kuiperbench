module Kuiper.KB.Softplus

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction
let softplus_step (#t:Type0) {| floating t |} (x : t) : t =
  flog (add one (fexp x))

inline_for_extraction noextract
type softplus_fw_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map softplus_step s)

val softplus_fw_f32 : softplus_fw_ty f32
val softplus_fw_f64 : softplus_fw_ty f64
