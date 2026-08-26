module Kuiper.KB.HardTanh

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let htanh_fw_f32 lena a #s = map_gpu htanh_step lena a
let htanh_fw_f64 lena a #s = map_gpu htanh_step lena a
