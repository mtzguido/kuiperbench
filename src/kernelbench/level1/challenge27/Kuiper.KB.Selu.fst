module Kuiper.KB.Selu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let selu_fw_f32 lena a #s = map_gpu selu_step lena a
let selu_fw_f64 lena a #s = map_gpu selu_step lena a
