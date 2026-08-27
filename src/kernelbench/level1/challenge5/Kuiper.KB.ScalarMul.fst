module Kuiper.KB.ScalarMul

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map
open Kuiper.Seq.Common

open Kuiper.Tensor.Layout.Alg { l1_forward }

let smul_fw_f32 c lena a #s = map_gpu (mul c) lena a
let smul_fw_f64 c lena a #s = map_gpu (mul c) lena a
let smul_fw_u32 c lena a #s = map_gpu (mul c) lena a
let smul_fw_u64 c lena a #s = map_gpu (mul c) lena a

inline_for_extraction noextract
fn smul_out_impl (#t:Type0) {| scalar t |}
  (cst : t)
  (lena : szp { lena <= max_blocks * max_threads })
  (c : array1 t (l1_forward lena) { is_global c })
  (a : array1 t (l1_forward lena) { is_global a })
  (#sc : chest1 t lena)
  (#sa : chest1 t lena)
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  requires  on gpu_loc (c |-> sc)
  ensures   on gpu_loc (c |-> chest_map (mul cst) sa)
{
  map_gpu2 (fun _xc xa -> mul cst xa) lena c a;
  assert (pure (equal (chest1_map2 (fun _xc xa -> mul cst xa) sc sa)
                      (chest_map (mul cst) sa)));
}

let smul_out_f32 = smul_out_impl
let smul_out_f64 = smul_out_impl
let smul_out_u32 = smul_out_impl
let smul_out_u64 = smul_out_impl
