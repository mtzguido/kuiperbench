module Kuiper.KB.HardSigmoid

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction
let hsig_step (#t:Type0) {| floating t |} (x : t) : t =
  let three  = of_int 3L in
  let mthree = sub zero three in
  let sixth  = div one (of_int 6L) in
  let half   = div one (of_int 2L) in
  if gte x three then one
  else if gte mthree x then zero
  else add (mul x sixth) half

inline_for_extraction noextract
type hsig_fw_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map hsig_step s)

val hsig_fw_f32 : hsig_fw_ty f32
val hsig_fw_f64 : hsig_fw_ty f64
