module Kuiper.KB.Softplus

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let softplus_fw_f32 lena a #s = map_gpu softplus_step lena a
let softplus_fw_f64 lena a #s = map_gpu softplus_step lena a
