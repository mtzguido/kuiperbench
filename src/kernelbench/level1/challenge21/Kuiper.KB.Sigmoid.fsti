module Kuiper.KB.Sigmoid

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
(* Pointwise sigmoid step: sigmoid(x) = 1 / (1 + exp(-x)). *)
inline_for_extraction
let sigmoid_step (#t:Type0) {| floating t |} (x : t) : t =
  one `div` (one `add` fexp (sub zero x))

inline_for_extraction noextract
type sigmoid_fw_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map sigmoid_step s)

(* Concrete row-major (forward) variants for extraction. *)
val sigmoid_fw_f32 : sigmoid_fw_ty f32
val sigmoid_fw_f64 : sigmoid_fw_ty f64

inline_for_extraction noextract
type sigmoid_alloc_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (input : array1 t (l1_forward lena) { is_global input })
     (#s : chest1 t lena)
     (#f : perm)
     norewrite
     preserves cpu ** on gpu_loc (input |-> Frac f s)
     returns output : array1 t (l1_forward lena)
     ensures on gpu_loc (output |-> mk1 (fun i -> sigmoid_step (acc1 s i)))

val sigmoid_alloc_f32 : sigmoid_alloc_ty f32
val sigmoid_alloc_f64 : sigmoid_alloc_ty f64
