module Kuiper.KB.Tanh

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let tanh_fw_f32 lena a #s = map_gpu tanh_step lena a
let tanh_fw_f64 lena a #s = map_gpu tanh_step lena a
