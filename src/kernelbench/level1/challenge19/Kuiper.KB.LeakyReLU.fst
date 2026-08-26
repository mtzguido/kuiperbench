module Kuiper.KB.LeakyReLU

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let leaky_relu_fw_f32 slope lena a #s = map_gpu (leaky_step slope) lena a
let leaky_relu_fw_f64 slope lena a #s = map_gpu (leaky_step slope) lena a
