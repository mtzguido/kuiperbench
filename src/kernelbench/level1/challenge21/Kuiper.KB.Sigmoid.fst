module Kuiper.KB.Sigmoid

(* KernelBench L1 #21: Sigmoid (in-place, layout-polymorphic).
   Body is a single call to the verified Kuiper.Kernel.Map.map_gpu. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let sigmoid_fw_f32 lena a #s = map_gpu sigmoid_step lena a
let sigmoid_fw_f64 lena a #s = map_gpu sigmoid_step lena a

inline_for_extraction noextract
fn sigmoid_alloc (#t:Type0) {| floating t |}
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 t (l1_forward lena) { is_global input })
  (#s : chest1 t lena)
  (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 t (l1_forward lena)
  ensures on gpu_loc (output |-> mk1 (fun i -> sigmoid_step (acc1 s i)))
{
  let output = alloc0 #t lena (l1_forward lena);
  map_gpu_to sigmoid_step lena input output;
  return output
}

let sigmoid_alloc_f32 = sigmoid_alloc #f32
let sigmoid_alloc_f64 = sigmoid_alloc #f64
