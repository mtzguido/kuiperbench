module Kuiper.KB.Swish

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let swish_fw_f32 lena a #s = map_gpu swish_step lena a
let swish_fw_f64 lena a #s = map_gpu swish_step lena a
