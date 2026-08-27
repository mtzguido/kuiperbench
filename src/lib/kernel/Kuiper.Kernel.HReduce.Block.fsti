module Kuiper.Kernel.HReduce.Block

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT

(* ── reduce_batched_block: one block per row, tree reduction in shmem ────────
   Sibling of [Kuiper.Kernel.HReduce.reduce_batched]. Spawns one block per row
   and performs a per-row tree reduction in shared memory. Lives in its own
   module so its (substantial) ghost surface doesn't perturb the SMT context
   of the legacy [HReduce] lemmas.
   ───────────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_block
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (#sx   : chest2 et   rows cols)
  (vr    : chest2 real rows cols)
  (#sout : chest1 et rows)
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (sx %~ vr)
  ensures
    exists* (sout' : chest1 et rows).
      on gpu_loc (output |-> sout') **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ rsum (Kuiper.Seq.Common.lseq_map pre_map_r (ematrix_row vr r)))
