module Kuiper.Math.Fmul

(* Narrow [*]-as-monoid axiomatisation for [f32].
 *
 * Mirror of [Kuiper.Math.Fadd] / [Kuiper.Math.Fmax] for the
 * cumprod-style scan primitives that consume [cmonoid_fmul_f32].
 * Bit-exact associativity *fails* for IEEE-754 [*], so this module is
 * consumed only by clients who lift to [%~]-style approximate
 * postconditions. *)

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Functions

val fmul_assoc       : squash (is_associative (mul #f32))
val fmul_comm        : squash (is_commutative (mul #f32))
val fmul_one_neutral : squash (is_neutral_for (one #f32) (mul #f32))
