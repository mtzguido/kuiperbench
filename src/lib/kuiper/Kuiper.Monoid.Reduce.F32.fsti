module Kuiper.Monoid.Reduce.F32

(* Concrete law-free reducers for the IEEE-754 operations used by pooling and
 * scans.  Their folds describe the exact operation order executed by the
 * kernel; clients may subsequently lift that result through [%~]. *)

open Kuiper.IntAliases
open Kuiper.Monoid.Reduce
open Kuiper.Scalars.Base

inline_for_extraction noextract
val reducer_fmax_f32 : reducer f32

inline_for_extraction noextract
val reducer_fadd_f32 : reducer f32

inline_for_extraction noextract
val reducer_fmul_f32 : reducer f32

val reducer_fadd_f32_proj (_:unit)
  : Lemma (reducer_fadd_f32.rop == (add #f32) /\
           reducer_fadd_f32.rid == (zero #f32))

val reducer_fmul_f32_proj (_:unit)
  : Lemma (reducer_fmul_f32.rop == (mul #f32) /\
           reducer_fmul_f32.rid == (one #f32))
