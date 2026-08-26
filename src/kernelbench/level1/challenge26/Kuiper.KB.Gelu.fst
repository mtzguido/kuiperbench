module Kuiper.KB.Gelu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let gelu_fw_f32 lena a #s = map_gpu gelu_step lena a
let gelu_fw_f64 lena a #s = map_gpu gelu_step lena a
