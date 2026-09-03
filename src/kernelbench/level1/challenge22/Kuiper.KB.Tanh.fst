module Kuiper.KB.Tanh

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let tanh_fw_f32 lena a #s = map_gpu tanh_step lena a
let tanh_fw_f64 lena a #s = map_gpu tanh_step lena a

inline_for_extraction noextract
fn tanh_alloc (#t:Type0) {| floating t |}
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 t (l1_forward lena) { is_global input })
  (#s : chest1 t lena) (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 t (l1_forward lena)
  ensures on gpu_loc (output |-> mk1 (fun i -> tanh_step (acc1 s i)))
{
  let output = alloc0 #t lena (l1_forward lena);
  map_gpu_to tanh_step lena input output;
  return output
}

let tanh_alloc_f32 = tanh_alloc #f32
let tanh_alloc_f64 = tanh_alloc #f64
