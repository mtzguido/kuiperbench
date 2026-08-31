module Kuiper.KB.TripletMarginLoss

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.TripletMarginLoss
module SZ = Kuiper.SizeT

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
    (ra rp rn : erased (lseq real (b * d)))
    (#fanc #fpos #fneg : perm)
    norewrite
    preserves cpu **
              on gpu_loc (anchor   |-> Frac fanc sa) **
              on gpu_loc (positive |-> Frac fpos sp) **
              on gpu_loc (negative |-> Frac fneg sn) **
              pure (sa %~ seq_to_chest1 ra /\
                    sp %~ seq_to_chest1 rp /\
                    sn %~ seq_to_chest1 rn)
    returns res : f32
    ensures
      pure (triplet_post b d margin eps ra rp rn res)
