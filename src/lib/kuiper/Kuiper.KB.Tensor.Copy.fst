module Kuiper.KB.Tensor.Copy

#lang-pulse

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT

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
{
  let dst = alloc0 #et n l;
  with sd. assert on gpu_loc (dst |-> sd);

  map_loc gpu_loc #(src |-> Frac f s) #(core src |-> Frac f (to_seq l s))
    fn _ { tensor_concr src; };
  map_loc gpu_loc #(dst |-> sd) #(core dst |-> to_seq l sd)
    fn _ { tensor_concr dst; };

  Kuiper.Array.Core.gpu_memcpy_device_to_device
    (core dst) (core src) n;

  (* [Tensor.Layout] exposes the two sequence conversions through the
     underlying full view; recover the logical chest after the raw copy. *)
  Kuiper.Tensor.Layout.from_seq_rel l (to_seq l s);
  Kuiper.Tensor.Layout.to_seq_rel l s;
  Kuiper.View.from_to (tensor_aview et l) s;
  assert pure (from_seq l (to_seq l s) == s);

  map_loc gpu_loc #(core src |-> Frac f (to_seq l s)) #(src |-> Frac f s)
    fn _ {
      tensor_abs l (core src);
      rewrite (from_array l (core src) |-> Frac f s)
           as (src |-> Frac f s);
    };
  map_loc gpu_loc #(core dst |-> to_seq l s) #(dst |-> s)
    fn _ {
      tensor_abs' l (core dst);
      rewrite (from_array l (core dst) |-> s)
           as (dst |-> s);
    };
  dst
}
