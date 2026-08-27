module Kuiper.KB.CumProd

(* KernelBench L1 #90: cumulative product (prefix product) along the
   inner dimension of a 2-D (B, D) row-major tensor.  PyTorch reference:
       y = torch.cumprod(x, dim=1)                   # shape (B, D)

   Same row-per-block sequential-scan kernel as CumSum, just specialised
   to the [cmonoid_fmul_f32] (multiplication / one) commutative monoid
   instead of [cmonoid_fadd_f32] (addition / zero).

   Postcondition lifts the bit-exact f32 product fold to an
   [%~]-approximation of the real-arithmetic ideal cumulative product
   ([rprod] over the corresponding slice of [to_real_seq] of the row).

   No assume / magic / admit.  Exactly 1 GPU kernel launch. *)

#lang-pulse
open Kuiper
open Kuiper.KB.Compat.Product
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Seq = FStar.Seq

let cumprod_post
  (#t : Type0) {| scalar t, real_like t |}
  (b d : nat)
  (sx : EM.chest2 t b d)
  (sy : EM.chest2 t b d)
  : prop
  = forall (r : nat) (i : nat).
      r < b /\ i < d ==>
      acc2 sy r i %~
        rprod (Seq.slice (to_real_seq (EM.ematrix_row sx r)) 0 (i + 1))

fn cumprod_fw_f32
  (b : szp { b <= max_blocks })
  (d : szp { SZ.fits (SZ.v b * SZ.v d) })
  (input  : array2 f32 (l2_row_major b d)
            { is_global input  })
  (output : array2 f32 (l2_row_major b d)
            { is_global output })
  (#sx #sy0 : EM.chest2 f32 b d)
  preserves
    cpu **
    on gpu_loc (input  |-> sx)
  requires
    on gpu_loc (output |-> sy0)
  ensures
    (exists* (sy : EM.chest2 f32 b d).
       on gpu_loc (output |-> sy) **
       pure (cumprod_post b d sx sy))


inline_for_extraction let () = ()
