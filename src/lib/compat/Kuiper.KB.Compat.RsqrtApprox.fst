module Kuiper.KB.Compat.RsqrtApprox

(* Temporary bridge for the reciprocal-square-root approximation law.  The
   selected Kuiper package provides [sqrt_approx] and [div_approx], but
   [rsqrt] is an independent floating-point primitive, so those laws do not
   characterize it.  Keep this local axiom while using the primitive, and
   delete the module when [rsqrt_approx] is packaged upstream. *)

open Kuiper
open Kuiper.Approximates

assume val rsqrt_approx
  (#a : Type0)
  {| scalar a, real_like a, floating a, floating_real_like a |}
  (x : a)
  (r : FStar.Math.Sqrt.rpos)
  : Lemma (requires x `v_approximates` r)
          (ensures rsqrt x `v_approximates` FStar.Math.Sqrt.rsqrt r)
