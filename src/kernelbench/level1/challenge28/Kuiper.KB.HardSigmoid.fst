module Kuiper.KB.HardSigmoid

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let hsig_fw_f32 lena a #s = map_gpu hsig_step lena a
let hsig_fw_f64 lena a #s = map_gpu hsig_step lena a
