module Kuiper.KB.Selu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
(* SELU constants (Klambauer et al., 2017), as literals. Each decimal is
   parsed into a C floating constant of the element type at extraction and
   inlined at the use site, so there is no runtime recomputation. *)
inline_for_extraction noextract
let selu_alpha  (#t:Type0) {| floating t |} : t = of_literal "1.6732632423543772848"
inline_for_extraction noextract
let selu_lambda (#t:Type0) {| floating t |} : t = of_literal "1.0507009873554804934"

inline_for_extraction
let selu_step (#t:Type0) {| floating t |} (x : t) : t =
  mul selu_lambda (if gt x zero then x else mul selu_alpha (sub (fexp x) one))

inline_for_extraction noextract
type selu_fw_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : chest1 t lena)
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map selu_step s)

val selu_fw_f32 : selu_fw_ty f32
val selu_fw_f64 : selu_fw_ty f64

inline_for_extraction noextract
type selu_alloc_ty (t:Type0) {| floating t |} =
  fn (lena : szp { lena <= max_blocks * max_threads })
     (input : array1 t (l1_forward lena) { is_global input })
     (#s : chest1 t lena)
     (#f : perm)
     norewrite
     preserves cpu ** on gpu_loc (input |-> Frac f s)
     returns output : array1 t (l1_forward lena)
     ensures on gpu_loc (output |-> mk1 (fun i -> selu_step (acc1 s i)))

val selu_alloc_f32 : selu_alloc_ty f32
val selu_alloc_f64 : selu_alloc_ty f64
