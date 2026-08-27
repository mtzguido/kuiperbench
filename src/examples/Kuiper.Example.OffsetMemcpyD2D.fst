module Kuiper.Example.OffsetMemcpyD2D

#lang-pulse

open Kuiper
module V = Pulse.Lib.Vec

fn main (_:unit)
  preserves cpu
  returns _ : u64
{
  let src = V.alloc 0uL 8sz;
  V.(src.(0sz) <- 10uL);
  let ga = gpu_array_alloc #u64 8sz;
  let zeros = V.alloc 0uL 8sz;
  Kuiper.Array.gpu_memcpy_host_to_device ga zeros 8sz;
  V.free zeros;
  Kuiper.Array.gpu_memcpy_host_to_device' ga 0sz #8 src 0sz 8sz;
  V.free src;

  let gb = gpu_array_alloc #u64 8sz;
  let zb = V.alloc 0uL 8sz;
  Kuiper.Array.gpu_memcpy_host_to_device gb zb 8sz;
  V.free zb;
  Kuiper.KB.Compat.Array.gpu_memcpy_device_to_device' gb 1sz #8 ga 2sz 3sz;

  let dst = V.alloc 0uL 8sz;
  Kuiper.Array.gpu_memcpy_device_to_host dst gb 8sz;
  Kuiper.Array.gpu_array_free ga;
  Kuiper.Array.gpu_array_free gb;
  let r = V.(dst.(1sz));
  V.free dst;
  r
}
