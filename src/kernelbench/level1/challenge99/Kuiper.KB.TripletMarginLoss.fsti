module Kuiper.KB.TripletMarginLoss

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.TripletMarginLoss
module SZ = Kuiper.SizeT

(* Verified, extractable reciprocal 1/B as f32 (see .fst). *)
val triplet_recip_f32 (b : szp) : f32

inline_for_extraction noextract
type triplet_fw_ty =
  fn
    (b : szp { b <= max_blocks * max_threads /\
               SZ.fits (b + max_threads) })
    (d : szp { d <= max_blocks * max_threads /\
               SZ.fits (d + max_threads) /\
               SZ.fits (b * d) })
    (margin inv_b : f32)
    (anchor   : array1 f32 (l1_forward (b * d)) { is_global anchor })
    (positive : array1 f32 (l1_forward (b * d)) { is_global positive })
    (negative : array1 f32 (l1_forward (b * d)) { is_global negative })
    (#sa #sp #sn : chest1 f32 (b * d))
    (#fanc #fpos #fneg : perm)
    preserves cpu **
              on gpu_loc (anchor   |-> Frac fanc sa) **
              on gpu_loc (positive |-> Frac fpos sp) **
              on gpu_loc (negative |-> Frac fneg sn)
    returns res : f32
    ensures
      pure (triplet_post (SZ.v b) (SZ.v d) margin inv_b
              (chest1_to_seq (reveal sa) <: Seq.lseq f32 (SZ.v b * SZ.v d))
              (chest1_to_seq (reveal sp) <: Seq.lseq f32 (SZ.v b * SZ.v d))
              (chest1_to_seq (reveal sn) <: Seq.lseq f32 (SZ.v b * SZ.v d))
              res)

val triplet_fw_f32 : triplet_fw_ty
