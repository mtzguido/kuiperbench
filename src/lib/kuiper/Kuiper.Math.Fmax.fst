module Kuiper.Math.Fmax

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating
open Kuiper.Float32
open Kuiper.Functions

let fmax_assoc : squash (is_associative (fmax #f32)) = admit()
let fmax_comm : squash (is_commutative (fmax #f32)) = admit()
let fmax_neg_inf_neutral
  : squash (is_neutral_for neg_inf (fmax #f32))
  = admit()
