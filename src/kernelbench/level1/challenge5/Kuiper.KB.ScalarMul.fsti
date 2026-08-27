module Kuiper.KB.ScalarMul

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
inline_for_extraction noextract
type smul_fw_ty (t:Type0) {| scalar t |} =
  fn (c : t)
     (lena : szp { lena <= max_blocks * max_threads })
     (a : array1 t (l1_forward lena) { is_global a })
     (#s : erased (chest1 t lena))
     preserves cpu
     requires  on gpu_loc (a |-> s)
     ensures   on gpu_loc (a |-> chest_map (mul c) s)

val smul_fw_f32 : smul_fw_ty f32
val smul_fw_f64 : smul_fw_ty f64
val smul_fw_u32 : smul_fw_ty u32
val smul_fw_u64 : smul_fw_ty u64

(* Out-of-place scalar multiply: writes c := scalar * a, leaving the input
   array [a] untouched (held at fractional permission).  This avoids forcing
   the caller to clone the input when, as in PyTorch's [A * s], a fresh output
   tensor is expected. *)
inline_for_extraction noextract
type smul_out_ty (t:Type0) {| scalar t |} =
  fn (cst : t)
     (lena : szp { lena <= max_blocks * max_threads })
     (c : array1 t (l1_forward lena) { is_global c })
     (a : array1 t (l1_forward lena) { is_global a })
     (#sc #sa : erased (chest1 t lena))
     (#fa : perm)
     preserves cpu ** on gpu_loc (a |-> Frac fa sa)
     requires  on gpu_loc (c |-> sc)
     ensures   on gpu_loc (c |-> chest_map (mul cst) sa)

val smul_out_f32 : smul_out_ty f32
val smul_out_f64 : smul_out_ty f64
val smul_out_u32 : smul_out_ty u32
val smul_out_u64 : smul_out_ty u64
