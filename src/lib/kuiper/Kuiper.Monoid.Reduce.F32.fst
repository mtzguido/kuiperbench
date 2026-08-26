module Kuiper.Monoid.Reduce.F32

(* Concrete [cmonoid f32] instances for [fmax] and [+].
 *
 * Both instances are [inline_for_extraction noextract] so that
 * Karamel sees concrete record literals at every consumer call
 * site, rather than abstract [val] symbols.  The carrier is the
 * abstract [Kuiper.Float32.t]; the operators (fmax, add) extract
 * to [fmaxf] / [+] respectively, and the identities ([neg_inf],
 * [zero]) extract to [(-INFINITY)] / [0.0f].
 *
 * The squashed monoid laws are pulled directly from the narrow
 * axiomatisations [Kuiper.Math.Fmax] / [Kuiper.Math.Fadd];
 * commutativity is derived from the [Kuiper.Float32.add_comm] /
 * [Kuiper.Math.Fmax.fmax_comm] SMT-pat lemmas. *)

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Monoid.Reduce
open Kuiper.Functions
module Fmax = Kuiper.Math.Fmax
module Fadd = Kuiper.Math.Fadd
module Fmul = Kuiper.Math.Fmul

inline_for_extraction noextract
let cmonoid_fmax_f32 : cmonoid f32 = {
  rid    = Fmax.neg_inf;
  rop    = fmax;
  rassoc = Fmax.fmax_assoc;
  rneut  = Fmax.fmax_neg_inf_neutral;
  rcomm  = Fmax.fmax_comm;
}

inline_for_extraction noextract
let cmonoid_fadd_f32 : cmonoid f32 = {
  rid    = zero;
  rop    = add;
  rassoc = Fadd.fadd_assoc;
  rneut  = Fadd.fadd_zero_neutral;
  rcomm  = Fadd.fadd_comm;
}

inline_for_extraction noextract
let cmonoid_fmul_f32 : cmonoid f32 = {
  rid    = one;
  rop    = mul;
  rassoc = Fmul.fmul_assoc;
  rneut  = Fmul.fmul_one_neutral;
  rcomm  = Fmul.fmul_comm;
}

let cmonoid_fadd_f32_proj ()
  : Lemma (cmonoid_fadd_f32.rop == (add #f32) /\
           cmonoid_fadd_f32.rid == (zero #f32))
  = ()

let cmonoid_fmul_f32_proj ()
  : Lemma (cmonoid_fmul_f32.rop == (mul #f32) /\
           cmonoid_fmul_f32.rid == (one #f32))
  = ()
