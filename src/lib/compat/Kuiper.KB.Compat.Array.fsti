module Kuiper.KB.Compat.Array

#lang-pulse

open Pulse
open FStar.Seq
open Kuiper
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT

(** Trusted extraction boundary for an offset-aware CUDA device-to-device copy.
    Its implementation is the function with the corresponding extracted name
    in [include/kbench.h].  Keeping [sized a] out of this foreign signature is
    intentional: C++ infers the element type from the pointer arguments.

    KaRaMeL retains placeholders for the erased [dst_sz], [src_sz], [f], [v],
    and [gv] arguments at calls to this polymorphic trusted function.  Their
    generated null-pointer values are expected and ignored by the C++ shim;
    only the arrays, offsets, and count are runtime inputs. *)
fn gpu_memcpy_device_to_device'
  (#a : Type u#0)
  (#dst_sz : erased nat)
  (dst_garr : larray a dst_sz)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_garr : larray a src_sz)
  (src_off : SZ.t)
  (cnt : SZ.t {
    dst_off + cnt <= dst_sz /\
    src_off + cnt <= src_sz
  })
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == src_sz })
  (#gv : erased (seq a) { Seq.length gv == dst_sz })
  preserves
    cpu **
    on gpu_loc (src_garr |-> Frac f (v <: seq _))
  requires
    on gpu_loc (dst_garr |-> gv)
  ensures
    exists* s'.
      on gpu_loc (dst_garr |-> s') **
      pure (s' == seq_blit gv dst_off v src_off cnt /\
            Seq.length s' == reveal dst_sz)
