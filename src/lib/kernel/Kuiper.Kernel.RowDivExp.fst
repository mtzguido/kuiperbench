module Kuiper.Kernel.RowDivExp

#lang-pulse

open Kuiper
module RB = Kuiper.Kernel.RowBroadcast

let row_div_exp
  (t : Type0) {| floating t, real_like t, floating_real_like t |}
  : row_div_exp_ty t
  = fun m n #_ #la #_ a #lb #_ b #_ #_ #fA #sa #sb ->
      RB.row_broadcast (div_exp #t) m n a b
