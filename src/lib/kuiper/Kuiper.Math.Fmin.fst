module Kuiper.Math.Fmin

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Functions

let fmin_assoc : squash (is_associative (fmin #f32)) = admit()
let fmin_comm : squash (is_commutative (fmin #f32)) = admit()
let fmin_pos_inf_neutral
  : squash (is_neutral_for pos_inf (fmin #f32))
  = admit()
