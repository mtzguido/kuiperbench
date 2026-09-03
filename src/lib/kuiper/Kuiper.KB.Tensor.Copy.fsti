module Kuiper.KB.Tensor.Copy

#lang-pulse

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT

(** Allocate a full-layout GPU tensor and copy the complete logical tensor
    into it.  The source is preserved fractionally and the returned tensor is
    owned by the caller.  This is the common verified boundary used by public
    out-of-place wrappers around otherwise in-place kernels. *)
inline_for_extraction noextract
fn copy_alloc
  (#et : Type) {| sized et |}
  (#r : nat) (#d : shape r)
  (n : szp { SZ.v n == sizeof d })
  (#l : tlayout d { is_full l })
  (src : tensor et l { is_global src })
  (#f : perm)
  (#s : chest d et)
  preserves
    cpu ** on gpu_loc (src |-> Frac f s)
  returns dst : tensor et l
  ensures
    on gpu_loc (dst |-> s) **
    pure (is_global dst) **
    pure (is_full_array (core dst))
