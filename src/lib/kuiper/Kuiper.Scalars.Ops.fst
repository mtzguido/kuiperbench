module Kuiper.Scalars.Ops

(* Reusable pointwise scalar operations.

   Lives in a separate module from [Kuiper.Scalars] so that adding
   helpers here does not change the SMT context of every module that
   only uses the [scalar] typeclass. *)

open Kuiper.Scalars

(* Pointwise square: [square x = x * x].

   Used by L1/L2-norm, Frobenius, MSE, BatchNorm / InstanceNorm /
   LayerNorm, etc. *)
inline_for_extraction
let square (#t:Type) {| scalar t |} (x : t) : t = mul x x
