module Kuiper.Spec.Pool2D

open Kuiper.Spec.Pool1D

let pool_out_dims_2d_zero h w kh kw sh sw ph pw dh dw =
  pool_out_len_1d_bound h kh sh ph dh;
  pool_out_len_1d_bound w kw sw pw dw
