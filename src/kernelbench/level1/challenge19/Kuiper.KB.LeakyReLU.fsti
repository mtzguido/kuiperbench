module Kuiper.KB.LeakyReLU

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Float.Casts
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction
let leaky_step (#t:Type0) {| floating t |} (slope x : t) : t =
  if gt x zero then x else mul x slope

inline_for_extraction noextract
type leaky_relu_fw_ty (t:Type0) {| floating t |} =
  fn (slope : t)
     (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map (leaky_step slope) s)

val leaky_relu_fw_f32 : leaky_relu_fw_ty f32
val leaky_relu_fw_f64 : leaky_relu_fw_ty f64

fn leaky_relu_alloc_f64_f32
  (slope : f64)
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 f32 (l1_forward lena) { is_global input })
  (#s : chest1 f32 lena)
  (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 f32 (l1_forward lena)
  ensures on gpu_loc
    (output |-> mk1 (fun i -> leaky_step (fcast slope) (acc1 s i)))

fn relu_alloc_f32
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 f32 (l1_forward lena) { is_global input })
  (#s : chest1 f32 lena)
  (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 f32 (l1_forward lena)
  ensures on gpu_loc
    (output |-> mk1 (fun i -> leaky_step (zero #f32) (acc1 s i)))
