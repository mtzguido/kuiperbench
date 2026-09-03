module Kuiper.KB.Elu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let elu_fw_f32 lena a #s = map_gpu elu_step lena a
let elu_fw_f64 lena a #s = map_gpu elu_step lena a

inline_for_extraction noextract
fn elu_alloc (#t:Type0) {| floating t |}
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 t (l1_forward lena) { is_global input })
  (#s : chest1 t lena) (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 t (l1_forward lena)
  ensures on gpu_loc (output |-> mk1 (fun i -> elu_step (acc1 s i)))
{
  let output = alloc0 #t lena (l1_forward lena);
  map_gpu_to elu_step lena input output;
  return output
}

let elu_alloc_f32 = elu_alloc #f32
let elu_alloc_f64 = elu_alloc #f64
