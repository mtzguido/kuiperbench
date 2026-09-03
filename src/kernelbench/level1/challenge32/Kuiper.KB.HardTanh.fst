module Kuiper.KB.HardTanh

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let htanh_fw_f32 lena a #s = map_gpu htanh_step lena a
let htanh_fw_f64 lena a #s = map_gpu htanh_step lena a

inline_for_extraction noextract
fn htanh_alloc (#t:Type0) {| floating t |}
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 t (l1_forward lena) { is_global input })
  (#s : chest1 t lena) (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 t (l1_forward lena)
  ensures on gpu_loc (output |-> mk1 (fun i -> htanh_step (acc1 s i)))
{
  let output = alloc0 #t lena (l1_forward lena);
  map_gpu_to htanh_step lena input output;
  return output
}

let htanh_alloc_f32 = htanh_alloc #f32
let htanh_alloc_f64 = htanh_alloc #f64
