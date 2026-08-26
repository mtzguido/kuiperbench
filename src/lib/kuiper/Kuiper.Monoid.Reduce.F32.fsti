module Kuiper.Monoid.Reduce.F32

(* Axiomatic [cmonoid f32] instances for IEEE-754 [fmax] and [+].
 *
 * Interface-only, mirroring [Kuiper.Approximates.F32].  These
 * declarations encode the *floating-point as a commutative monoid*
 * convention used throughout the Kuiper stack: associativity and
 * commutativity are postulated for f32 fadd / fmax at the
 * type-class instance boundary so that downstream kernel proofs
 * can reorder partial reductions.  Strict bit-exact violations of
 * those laws on FP are accounted for by [%~] postconditions in
 * the corresponding kernel specs (e.g. [avgpool1d_post]). *)

open Kuiper.IntAliases
open Kuiper.Monoid.Reduce

inline_for_extraction noextract
val cmonoid_fmax_f32 : cmonoid f32

inline_for_extraction noextract
val cmonoid_fadd_f32 : cmonoid f32

inline_for_extraction noextract
val cmonoid_fmul_f32 : cmonoid f32

(* Component-level equalities exposed to clients so that proofs which
   need to unfold [red_fold cmonoid_fadd_f32 ...] into [seq_fold_left
   add zero ...] can do so without breaking the abstraction barrier
   on the cmonoid record itself. *)
open Kuiper.Scalars.Base
val cmonoid_fadd_f32_proj (_:unit)
  : Lemma (cmonoid_fadd_f32.rop == (add #f32) /\
           cmonoid_fadd_f32.rid == (zero #f32))

val cmonoid_fmul_f32_proj (_:unit)
  : Lemma (cmonoid_fmul_f32.rop == (mul #f32) /\
           cmonoid_fmul_f32.rid == (one #f32))
