module Kuiper.KB.ArgmaxReduceDim

(* KernelBench L1 #51: argmax over the middle dimension of a (B, D, M)
   row-major tensor.  PyTorch reference:
       y = torch.argmax(x, dim=1)         # shape (B, M), dtype int64

   Kuiper view: factor (B, D, M) as a 2-D matrix of shape (B*M, D)
   using Kuiper.Tensor.Layout.BCMPages.l2_bcm_pages.  Row r = b*M + j
   carries the length-D slice x[b, :, j].  One launch of
   Kuiper.Kernel.HReduce.Argmax.reduce_batched_argmax_f32 produces the
   exact index computed by a left-to-right scan seeded with (0, -inf)
   and updated only when [Float32.gt] returns true.  This operational
   specification is deterministic and non-vacuous without postulating
   a relationship between Kuiper's abstract [gt] and [fmax].

   Exactly 1 GPU kernel launch. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.BCMPages
module SZ = Kuiper.SizeT
module HRA = Kuiper.Kernel.HReduce.Argmax
module Seq = FStar.Seq
module I64 = FStar.Int64

(* Per-row post: the i64 output is in-bounds and equals the result of
   the strict-comparison scan used by the kernel. *)
let argmaxreduce_post
  (n_rows : nat) (n_cols : nat{n_cols > 0})
  (sx : chest2 f32 n_rows n_cols)
  (sy : Seq.lseq I64.t n_rows)
  : prop =
  forall (r : nat). r < n_rows ==>
    (let bi = I64.v (Seq.index sy r) in
     0 <= bi /\ bi < n_cols /\
     bi == fst (HRA.row_argmax_partial sx r n_cols))

fn argmaxreduce_dim_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads /\
             SZ.v d < pow2 63 })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (y : array1 I64.t (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  (#sy : chest1 I64.t (SZ.v b * SZ.v m))
  preserves cpu ** on gpu_loc (x |-> sx)
  requires on gpu_loc (y |-> sy)
  ensures
    exists* (sy' : chest1 I64.t (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy') **
      pure (argmaxreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy'))

fn argmaxreduce_dim_alloc_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks * max_threads /\
             SZ.v d < pow2 63 })
  (x : array2 f32 (l2_bcm_pages b m d) { is_global x })
  (#sx : chest2 f32 (SZ.v b * SZ.v m) d)
  norewrite
  preserves cpu ** on gpu_loc (x |-> sx)
  returns y : array1 I64.t (l1_forward (SZ.v b * SZ.v m))
  ensures
    exists* (sy : chest1 I64.t (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy) **
      pure (argmaxreduce_post (SZ.v b * SZ.v m) d sx (chest1_to_seq sy))
