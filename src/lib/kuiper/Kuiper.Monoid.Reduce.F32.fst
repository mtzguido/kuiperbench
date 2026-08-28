module Kuiper.Monoid.Reduce.F32

open Kuiper.IntAliases
open Kuiper.Scalars.Base
open Kuiper.Floating.Base
open Kuiper.Float32
open Kuiper.Monoid.Reduce
module Fmax = Kuiper.Math.Fmax

inline_for_extraction noextract
let reducer_fmax_f32 : reducer f32 = {
  rid = Fmax.neg_inf;
  rop = fmax;
}

inline_for_extraction noextract
let reducer_fadd_f32 : reducer f32 = {
  rid = zero;
  rop = add;
}

inline_for_extraction noextract
let reducer_fmul_f32 : reducer f32 = {
  rid = one;
  rop = mul;
}

let reducer_fadd_f32_proj ()
  : Lemma (reducer_fadd_f32.rop == (add #f32) /\
           reducer_fadd_f32.rid == (zero #f32))
  = ()

let reducer_fmul_f32_proj ()
  : Lemma (reducer_fmul_f32.rop == (mul #f32) /\
           reducer_fmul_f32.rid == (one #f32))
  = ()
