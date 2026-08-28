module Kuiper.Monoid.Reduce

(* A reduction operator with a seed.
 *
 * Unlike the former commutative-monoid abstraction, this record deliberately carries
 * no algebraic laws.  Sequential folds, window reductions, and the current
 * scan kernels execute their operations in a fixed left-to-right order, so
 * associativity, commutativity, and neutrality are neither true for IEEE-754
 * operations nor needed by their proofs. *)

open Kuiper.IntAliases

noeq
type reducer (t : Type) = {
  rid : t;
  rop : t -> t -> t;
}

(* The implementation-order fold used by the verified kernels. *)
let red_fold (#t:Type) (r : reducer t) (acc : t) (s : FStar.Seq.seq t)
  : GTot t
  = Kuiper.Seq.Common.seq_fold_left r.rop acc s
