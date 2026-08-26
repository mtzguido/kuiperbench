module Kuiper.Math.Fadd

(* Narrow [+]-as-monoid axiomatisation for [f32].
 *
 * Mirror of [Kuiper.Math.Fmax] for the avg-pool / sum-reduction
 * primitives that consume [cmonoid_fadd_f32].  Bit-exact
 * associativity *fails* for IEEE-754 [+], so this module is
 * consumed only by clients who lift to [%~]-style approximate
 * postconditions.  It is exposed as an axiomatic instance in
 * the same idiom as [Kuiper.Math.Fmax] / [Kuiper.Approximates.F32]:
 * stated at the type-class instance boundary, justified by the
 * surrounding [%~] post. *)

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Functions

val fadd_assoc        : squash (is_associative (add #f32))
val fadd_comm         : squash (is_commutative (add #f32))
val fadd_zero_neutral : squash (is_neutral_for (zero #f32) (add #f32))
