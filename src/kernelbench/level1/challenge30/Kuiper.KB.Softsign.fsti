module Kuiper.KB.Softsign

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction
let softsign_step (#t:Type0) {| floating t |} (x : t) : t = div x (add one (fabs x))

inline_for_extraction noextract
type softsign_fw_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map softsign_step s)

val softsign_fw_f32 : softsign_fw_ty f32
val softsign_fw_f64 : softsign_fw_ty f64

inline_for_extraction noextract
type softsign_alloc_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (input : array1 t (l1_forward lena) { is_global input })
     (#s : chest1 t lena)
     (#f : perm)
     norewrite
     preserves cpu ** on gpu_loc (input |-> Frac f s)
     returns output : array1 t (l1_forward lena)
     ensures on gpu_loc (output |-> mk1 (fun i -> softsign_step (acc1 s i)))

val softsign_alloc_f32 : softsign_alloc_ty f32
val softsign_alloc_f64 : softsign_alloc_ty f64
