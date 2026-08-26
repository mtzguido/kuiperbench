module Kuiper.KB.NewGelu

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let newgelu_fw_f32 half c k lena a #s = map_gpu (newgelu_step half c k) lena a
let newgelu_fw_f64 half c k lena a #s = map_gpu (newgelu_step half c k) lena a
