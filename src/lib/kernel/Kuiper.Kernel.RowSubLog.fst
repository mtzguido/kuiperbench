module Kuiper.Kernel.RowSubLog

#lang-pulse

open Kuiper
module RB = Kuiper.Kernel.RowBroadcast

let row_sub_log
  (t : Type0) {| floating t, real_like t, floating_real_like t |}
  : row_sub_log_ty t
  = fun m n #_ #la #_ a #lb #_ b #_ #_ #fA #sa #sb ->
      RB.row_broadcast (sub_log #t) m n a b
