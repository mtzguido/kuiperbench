module Kuiper.KB.Gelu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let gelu_fw_f32 lena a #s = map_gpu gelu_step lena a
let gelu_fw_f64 lena a #s = map_gpu gelu_step lena a

inline_for_extraction noextract
fn gelu_alloc (#t:Type0) {| floating t |}
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 t (l1_forward lena) { is_global input })
  (#s : chest1 t lena) (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 t (l1_forward lena)
  ensures on gpu_loc (output |-> mk1 (fun i -> gelu_step (acc1 s i)))
{
  let output = alloc0 #t lena (l1_forward lena);
  map_gpu_to gelu_step lena input output;
  return output
}

let gelu_alloc_f32 = gelu_alloc #f32
let gelu_alloc_f64 = gelu_alloc #f64
