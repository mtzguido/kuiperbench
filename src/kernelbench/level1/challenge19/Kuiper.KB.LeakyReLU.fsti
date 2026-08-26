module Kuiper.KB.LeakyReLU

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction
let leaky_step (#t:Type0) {| floating t |} (slope x : t) : t =
  if gt x zero then x else mul x slope

inline_for_extraction noextract
type leaky_relu_fw_ty (t:Type0) {| floating t |} =
  fn (slope : t)
     (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : erased (chest1 t lena))
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map (leaky_step slope) s)

val leaky_relu_fw_f32 : leaky_relu_fw_ty f32
val leaky_relu_fw_f64 : leaky_relu_fw_ty f64
