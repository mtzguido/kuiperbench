module Kuiper.KB.Compat.SqrtApprox

(* Temporary bridge for the square-root approximation law proposed for
   Kuiper.Approximates.floating_real_like.  Delete this module and use the
   packaged definition once the accompanying Kuiper patch is released. *)

open Kuiper
open Kuiper.Approximates

assume val sqrt_approx
  (#a : Type0)
  {| scalar a, real_like a, floating a, floating_real_like a |}
  (x : a)
  (r : FStar.Math.Sqrt.rnonneg)
  : Lemma (requires x `v_approximates` r)
          (ensures sqrt x `v_approximates` FStar.Math.Sqrt.sqrt r)
