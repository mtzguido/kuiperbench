module Kuiper.KB.LeakyReLU

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map
open Kuiper.Float.Casts

open Kuiper.Tensor.Layout.Alg { l1_forward }

let leaky_relu_fw_f32 slope lena a #s = map_gpu (leaky_step slope) lena a
let leaky_relu_fw_f64 slope lena a #s = map_gpu (leaky_step slope) lena a

inline_for_extraction noextract
fn leaky_relu_alloc (#t:Type0) {| floating t |}
  (slope : t)
  (lena : szp { lena <= max_blocks * max_threads })
  (input : array1 t (l1_forward lena) { is_global input })
  (#s : chest1 t lena)
  (#f : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac f s)
  returns output : array1 t (l1_forward lena)
  ensures on gpu_loc
    (output |-> mk1 (fun i -> leaky_step slope (acc1 s i)))
{
  let output = alloc0 #t lena (l1_forward lena);
  map_gpu_to (leaky_step slope) lena input output;
  return output
}

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
{
  leaky_relu_alloc #f32 (fcast slope) lena input
}

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
{
  leaky_relu_alloc #f32 zero lena input
}
