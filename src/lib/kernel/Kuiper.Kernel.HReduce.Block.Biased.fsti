module Kuiper.Kernel.HReduce.Block.Biased

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor
module SZ = Kuiper.SizeT
open Kuiper.Kernel.HReduce {} (* for the [approx_function_can_approximate] instance *)

(* ── reduce_batched_block_biased: biased batched block reduction ──────────
   Like [reduce_batched_block] but takes an additional 1D bias array. Each
   row's reduction computes [sum_j pre_map (x[i,j]) (bias[i])].
   ───────────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_block_biased
  (#et : Type0) {| scalar et, real_like et |}
  (pre_map : et -> et -> et)
  (pre_map_r : real -> real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth  : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin   : layout2 (SZ.v rows) (SZ.v cols)) {| ctlayout lin   |}
  (#lbias : layout1 (SZ.v rows))             {| ctlayout lbias |}
  (#lout  : layout1 (SZ.v rows))             {| ctlayout lout  |}
  (x      : array2 et lin   { is_global x      })
  (bias   : array1 et lbias { is_global bias   })
  (output : array1 et lout  { is_global output })
  (#sx    : chest2 et   (SZ.v rows) (SZ.v cols))
  (vr     : chest2 real (SZ.v rows) (SZ.v cols))
  (#sbias : erased (chest1 et (SZ.v rows)))
  (vbias  : erased (chest1 real (SZ.v rows)))
  (#sout  : erased (chest1 et (SZ.v rows)))
  (#fbias : perm)
  preserves
    cpu **
    on gpu_loc (x |-> sx) **
    on gpu_loc (bias |-> Frac fbias sbias)
  requires
    on gpu_loc (output |-> sout) **
    pure (sx %~ vr) **
    pure (sbias %~ vbias)
  ensures
    exists* (sout' : chest1 et (SZ.v rows)).
      on gpu_loc (output |-> sout') **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ rsum (Kuiper.Seq.Common.lseq_map (fun c -> pre_map_r c (acc1 vbias r))
                                                             (ematrix_row vr r)))
