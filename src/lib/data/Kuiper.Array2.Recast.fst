module Kuiper.Array2.Recast

(* Reinterpret an Array2 buffer through a different (same-size) full
   layout, sharing the underlying backing array -- no data movement.

   This is the runtime-free "view cast" used by the separable 2-D pool:
   the row-major (bc*H, W_out) intermediate produced by pass 1 is recast
   to the [flat_bcm bc W_out H] view so pass 2 reduces the H axis
   strided.  Since [T.array2 et l = tensor backed by core a : larray et
   (layout_size l)] and both layouts are full (so [layout_size = rows*cols]),
   when the element counts are provably equal the backing array is reused
   verbatim. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor.Layout { is_full, full_layout_size, from_seq, to_seq, layout2 }
module T = Kuiper.Tensor
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn recast
  (#et : Type0)
  (#r1 #c1 #r2 #c2 : nat)
  (#l1 : layout2 r1 c1 { is_full l1 })
  (l2 : layout2 r2 c2 { is_full l2 })
  (a1 : T.array2 et l1)
  (#_ : squash (r1 * c1 == r2 * c2))
  (#f : perm)
  (#s1 : Ghost.erased (EM.chest2 et r1 c1))
  requires
    (a1 |-> Frac f s1)
  returns a2 : T.array2 et l2
  ensures
    (a2 |-> Frac f (from_seq l2 (to_seq l1 s1))) **
    pure (T.is_global a2 <==> T.is_global a1) **
    pure (is_full_array (T.core a1) ==> is_full_array (T.core a2))
{
  full_layout_size l1;
  full_layout_size l2;
  T.tensor_concr #et #_ #_ #l1 a1;
  let p2 : larray et (T.tlayout_ulen l2) = T.core a1;
  rewrite (T.core a1 |-> Frac f (to_seq l1 s1))
       as (p2 |-> Frac f (to_seq l1 s1));
  T.tensor_abs' #et #_ #_ l2 p2 #f #(to_seq l1 s1);
  let a2 = T.from_array l2 p2;
  rewrite (T.from_array l2 p2 |-> Frac f (from_seq l2 (to_seq l1 s1)))
       as (a2 |-> Frac f (from_seq l2 (to_seq l1 s1)));
  a2
}

(* The pure (runtime, zero-cost) view value produced by a recast: the SAME
   backing array, reinterpreted through [l2].  [from_array]/[core] are both
   inline_for_extraction pointer reinterpretations, so this compiles to a
   no-op pointer copy. *)
inline_for_extraction noextract
let recast_view
  (#et : Type0)
  (#r1 #c1 #r2 #c2 : nat)
  (#l1 : layout2 r1 c1)
  (l2 : layout2 r2 c2)
  (a1 : T.array2 et l1)
  (#_ : squash (T.tlayout_ulen l1 == T.tlayout_ulen l2))
  : T.array2 et l2
  = T.from_array l2 (T.core a1)

(* Located recast: reshape a GPU-resident buffer's points-to predicate
   (which lives under [on gpu_loc ...]) from layout [l1] to the equal-size
   full layout [l2], sharing the backing array, and return the reinterpreted
   handle [a2].  Uses the [ghost_impersonate] pattern (cf.
   Kuiper.Matrix.gpu_matrix_pts_to_ref_located): the located pre/post are
   [placeless] (via [placeless_on]); inside the impersonation we [on_elim] to
   expose the bare predicate, run the ghost [lower]/[raise'] reshape, and
   [on_intro] the result.  No data is moved (compiles to a pointer copy). *)
inline_for_extraction noextract
fn recast_gpu
  (#et : Type0)
  (#r1 #c1 #r2 #c2 : nat)
  (#l1 : layout2 r1 c1 { is_full l1 })
  (l2 : layout2 r2 c2 { is_full l2 })
  (a1 : T.array2 et l1)
  (#_ : squash (r1 * c1 == r2 * c2))
  (#loc : loc_id)
  (#f : perm)
  (#s1 : Ghost.erased (EM.chest2 et r1 c1))
  requires
    on loc (a1 |-> Frac f s1) **
    pure (T.is_global a1) **
    pure (is_full_array (T.core a1))
  returns a2 : T.array2 et l2
  ensures
    on loc (a2 |-> Frac f (from_seq l2 (to_seq l1 s1))) **
    pure (T.core a2 == T.core a1) **
    pure (T.is_global a2) **
    pure (is_full_array (T.core a2))
{
  full_layout_size l1;
  full_layout_size l2;
  let a2 = recast_view #et #r1 #c1 #r2 #c2 #l1 l2 a1 #();
  ghost_impersonate loc
    (on loc (a1 |-> Frac f s1))
    (on loc (a2 |-> Frac f (from_seq l2 (to_seq l1 s1))))
    fn () {
      on_elim _;
      T.tensor_concr #et #_ #_ #l1 a1;
      let p2 : larray et (T.tlayout_ulen l2) = T.core a1;
      rewrite (T.core a1 |-> Frac f (to_seq l1 s1))
           as (p2 |-> Frac f (to_seq l1 s1));
      T.tensor_abs' #et #_ #_ l2 p2 #f #(to_seq l1 s1);
      rewrite (T.from_array l2 p2
                 |-> Frac f (from_seq l2 (to_seq l1 s1)))
           as (a2 |-> Frac f (from_seq l2 (to_seq l1 s1)));
      on_intro (a2 |-> Frac f (from_seq l2 (to_seq l1 s1)));
    };
  a2
}
