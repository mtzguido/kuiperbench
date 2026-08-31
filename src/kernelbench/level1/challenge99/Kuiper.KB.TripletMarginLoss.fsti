module Kuiper.KB.TripletMarginLoss

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.TripletMarginLoss
module SZ = Kuiper.SizeT

(* Verified reciprocal 1/B as f32. It is inlined into [triplet_fw_f32]
   and is not part of the extracted C ABI. *)
inline_for_extraction noextract
val triplet_recip_f32 (b : szp)
  : r:f32 {
      r == div one (of_int (FStar.Int.Cast.uint64_to_int64
                              (FStar.SizeT.sizet_to_uint64 b)))
    }

fn triplet_fw_f32
    (b : szp { b <= max_blocks * max_threads /\
               SZ.fits (b + max_threads) })
    (d : szp { d <= max_blocks * max_threads /\
               SZ.fits (d + max_threads) /\
               SZ.fits (b * d) })
    (margin eps : f32)
    (anchor   : array1 f32 (l1_forward (b * d)) { is_global anchor })
    (positive : array1 f32 (l1_forward (b * d)) { is_global positive })
    (negative : array1 f32 (l1_forward (b * d)) { is_global negative })
    (#sa #sp #sn : chest1 f32 (b * d))
    (#fanc #fpos #fneg : perm)
    norewrite
    preserves cpu **
              on gpu_loc (anchor   |-> Frac fanc sa) **
              on gpu_loc (positive |-> Frac fpos sp) **
              on gpu_loc (negative |-> Frac fneg sn)
    returns res : f32
    ensures
      pure (triplet_post b d margin eps (triplet_recip_f32 b)
              (chest1_to_seq sa)
              (chest1_to_seq sp)
              (chest1_to_seq sn)
              res)
