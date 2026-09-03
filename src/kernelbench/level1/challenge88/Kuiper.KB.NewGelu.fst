module Kuiper.KB.NewGelu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let newgelu_fw_f32 half c k lena a #s = map_gpu (newgelu_step half c k) lena a
let newgelu_fw_f64 half c k lena a #s = map_gpu (newgelu_step half c k) lena a

inline_for_extraction noextract
fn newgelu_alloc (#t:Type0) {| floating t |}
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 t (l1_forward lena) { is_global input })
  (#s : chest1 t lena) (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 t (l1_forward lena)
  ensures on gpu_loc
    (output |-> mk1 (fun i -> newgelu_default_step (acc1 s i)))
{
  let output = alloc0 #t lena (l1_forward lena);
  map_gpu_to newgelu_default_step lena input output;
  return output
}

let newgelu_alloc_f32 = newgelu_alloc #f32
let newgelu_alloc_f64 = newgelu_alloc #f64
