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
