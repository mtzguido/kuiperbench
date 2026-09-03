module Kuiper.KB.TripletMarginLoss

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.TripletMarginLoss
open Kuiper.Float.Casts
module SZ = Kuiper.SizeT

inline_for_extraction
let triplet_default_eps_f32 : f32 = of_literal "0.000001"

fn triplet_fw_f32
    (b : szp { b <= max_blocks * max_threads /\
               SZ.fits (b + max_threads) })
    (d : szp { d <= max_blocks * max_threads /\
               SZ.fits (d + max_threads) /\
               SZ.fits (b * d) })
    (margin : f64)
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
    returns out : array1 f32 (l1_forward 1)
    ensures
      exists* (sout : chest1 f32 1).
        on gpu_loc (out |-> sout) **
        pure (triplet_post b d (fcast margin) triplet_default_eps_f32
                ra rp rn (acc1 sout 0))
