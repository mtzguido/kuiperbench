module Kuiper.KB.Softsign

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.Map

open Kuiper.Tensor.Layout.Alg { l1_forward }

let softsign_fw_f32 lena a #s = map_gpu softsign_step lena a
let softsign_fw_f64 lena a #s = map_gpu softsign_step lena a
