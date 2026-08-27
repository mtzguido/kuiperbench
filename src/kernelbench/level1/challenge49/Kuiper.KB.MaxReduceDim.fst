module Kuiper.KB.MaxReduceDim

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Math.OnlineSoftmax { seq_max }
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Layout.BCMPages
open Kuiper.Spec.MaxReduceDim
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module BMax = Kuiper.Kernel.HReduce.Block.Max
module KS = Kuiper.Seq.Common
module Seq = FStar.Seq

(* Clamp the block thread count so it never exceeds the data length: this keeps
   every strided bucket of the max reduction non-empty (max has no real-number
   identity).  Local copy of [Kuiper.Kernel.Softmax.clamp_threads], which is not
   exported. *)
inline_for_extraction noextract
let clamp_threads (nth lena : szp)
  : (r : szp { SZ.v r <= SZ.v nth /\ SZ.v r <= SZ.v lena })
  = if nth <=^ lena then nth else lena

(* Bridge lemma: a row of [EM.to_real_matrix sx] equals [to_real_seq] of
   the corresponding row of [sx], as sequences.  Both sides are
   length-[cols] sequences whose [j]-th element is [to_real (acc2 sx r j)]. *)
let row_to_real_eq
  (#t:Type0) {| scalar t, real_like t |}
  (#rows #cols : nat)
  (sx : EM.chest2 t rows cols)
  (r : nat { r < rows })
  : Lemma (Seq.equal
             (EM.ematrix_row (EM.to_real_matrix sx) r)
             (to_real_seq (EM.ematrix_row sx r)))
  = let lhs = EM.ematrix_row (EM.to_real_matrix sx) r in
    let rhs = to_real_seq (EM.ematrix_row sx r) in
    let aux (j:nat{j<cols}) : Lemma (Seq.index lhs j == Seq.index rhs j) = () in
    Classical.forall_intro aux

(* Per-row simplification of the [reduce_batched_block_max] postcondition
   into the form used by [maxreduce_post]:
       seq_max (lseq_map id (ematrix_row (to_real_matrix sx) r))
     = seq_max (to_real_seq (ematrix_row sx r)). *)
let row_post_eq
  (#t:Type0) {| scalar t, real_like t |}
  (#rows : nat) (#cols : nat{cols > 0})
  (sx : EM.chest2 t rows cols)
  (r : nat { r < rows })
  : Lemma
      (seq_max (KS.lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r))
       == seq_max (to_real_seq (EM.ematrix_row sx r)))
  = row_to_real_eq sx r;
    Seq.lemma_eq_intro
      (KS.lseq_map id (EM.ematrix_row (EM.to_real_matrix sx) r))
      (to_real_seq (EM.ematrix_row sx r))

(* [reduce_batched_block_max] states its per-row postcondition in the chest-native
   form [chest1_max (chest_map pre_map_r (chest2_row vr r))], whereas [maxreduce_post]
   is stated with [seq_max (to_real_seq (ematrix_row sx r))].  The following bridge
   the two.  They are local copies of the (unexported) helpers in
   [Kuiper.Kernel.HReduce.Block.Max]; [chest1_max_seq] and [seq_max] agree on the
   underlying non-empty seq, and [chest1_to_seq (chest_map f (chest2_row vr r))]
   is extensionally [lseq_map f (ematrix_row vr r)]. *)
let rec chest1_max_seq_is_seq_max (s : Seq.seq real { Seq.length s > 0 })
  : Lemma (ensures chest1_max_seq s == seq_max s) (decreases Seq.length s)
  = if Seq.length s = 1 then () else chest1_max_seq_is_seq_max (Seq.slice s 1 (Seq.length s))

let chest1_max_is_seq_max (#n : nat { n > 0 }) (c : chest1 real n)
  : Lemma (chest1_max c == seq_max (chest1_to_seq c))
  = chest1_max_seq_is_seq_max (chest1_to_seq c)

let max_row_bridge
  (pre_map_r : real -> real)
  (#rows : nat) (#cols : nat { cols > 0 })
  (vr : EM.chest2 real rows cols) (r : natlt rows)
  : Lemma (ensures seq_max (KS.lseq_map pre_map_r (EM.ematrix_row vr r))
                   == chest1_max (chest_map pre_map_r (chest2_row vr r)))
  = Seq.lemma_eq_elim (chest1_to_seq (chest_map pre_map_r (chest2_row vr r)))
                      (KS.lseq_map pre_map_r (EM.ematrix_row vr r));
    chest1_max_is_seq_max (chest_map pre_map_r (chest2_row vr r))

(* Transport an approximation across a real equality by pure congruence, keeping
   [a]/[b] abstract so Z3 does not unfold [%~] over the arithmetic-heavy
   [seq_max]/[chest1_max] terms. *)
let approximates_subst (x : f32) (a b : real)
  : Lemma (requires x %~ a /\ a == b) (ensures x %~ b) = ()

(* Turn the chest-native launch postcondition into [maxreduce_post]. *)
let maxreduce_post_from_chest
  (#rows : nat) (#cols : nat { cols > 0 })
  (sx : EM.chest2 f32 rows cols)
  (sy' : chest1 f32 rows)
  : Lemma
      (requires (forall (r : nat). r < rows ==>
                   acc1 sy' r %~ chest1_max (chest_map (id <: real -> real)
                                              (chest2_row (EM.to_real_matrix sx) r))))
      (ensures maxreduce_post rows cols sx (chest1_to_seq sy'))
  = introduce forall (r : nat). r < rows ==>
        (Seq.index (chest1_to_seq sy') r) %~ seq_max (to_real_seq (EM.ematrix_row sx r))
    with introduce _ ==> _
    with (
      max_row_bridge (id <: real -> real) #rows #cols (EM.to_real_matrix sx) r;
      row_post_eq sx r;
      approximates_subst (acc1 sy' r)
        (chest1_max (chest_map (id <: real -> real) (chest2_row (EM.to_real_matrix sx) r)))
        (seq_max (to_real_seq (EM.ematrix_row sx r)))
    )

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn maxreduce_dim_fw_f32_impl
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks /\
             SZ.fits (SZ.v d + max_threads) })
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v m) (SZ.v d)) { is_global x })
  (y : array1 f32 (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : EM.chest2 f32 (SZ.v b * SZ.v m) (SZ.v d))
  (#sy : chest1 f32 (SZ.v b * SZ.v m))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    on gpu_loc (y |-> sy)
  ensures
    on gpu_loc (x |-> sx) **
    (exists* (sy' : chest1 f32 (SZ.v b * SZ.v m)).
       on gpu_loc (y |-> sy') **
       pure (maxreduce_post (SZ.v b * SZ.v m) (SZ.v d) sx (chest1_to_seq sy')))
{
  let bm : szp = b *^ m;
  assert pure (SZ.v bm == SZ.v b * SZ.v m);
  (* Build the real-valued ghost chest2 and the sx %~ vr witness. *)
  let vr : EM.chest2 real (SZ.v b * SZ.v m) (SZ.v d) =
    hide (EM.to_real_matrix (reveal sx));
  assert pure (reveal sx %~ reveal vr);
  let vr' : EM.chest2 real (SZ.v bm) (SZ.v d) = vr;
  (* Clamp the block thread count so every strided bucket is non-empty
     (max has no real-number identity, so [nth <= cols] is required). *)
  let nthm : szp = clamp_threads max_threads d;
  assert pure (SZ.fits (SZ.v d + SZ.v nthm));
  BMax.reduce_batched_block_max #f32 id id bm d nthm
    #_ #(c_l2_bcm_pages (SZ.v b) m d)
    #_ #(c_l1_forward _)
    x y vr';
  with sy'. assert (on gpu_loc (y |-> sy'));
  (* Bridge the chest-native per-row max postcondition into [maxreduce_post]. *)
  maxreduce_post_from_chest #(SZ.v bm) #(SZ.v d) (reveal sx) sy';
  ()
}
#pop-options

fn maxreduce_dim_fw_f32
  (b : szp)
  (m : SZ.t { 0 < SZ.v m /\ SZ.fits (SZ.v b * SZ.v m) })
  (d : szp { SZ.fits (SZ.v m * SZ.v d) /\
             SZ.fits (SZ.v b * (SZ.v m * SZ.v d)) /\
             SZ.v b * SZ.v m <= max_blocks /\
             SZ.fits (SZ.v d + max_threads) })
  (x : array2 f32 (l2_bcm_pages (SZ.v b) (SZ.v m) (SZ.v d)) { is_global x })
  (y : array1 f32 (l1_forward (SZ.v b * SZ.v m)) { is_global y })
  (#sx : EM.chest2 f32 (SZ.v b * SZ.v m) (SZ.v d))
  (#sy : chest1 f32 (SZ.v b * SZ.v m))
  preserves cpu ** on gpu_loc (x |-> sx)
  requires
    on gpu_loc (y |-> sy)
  ensures
    exists* (sy' : chest1 f32 (SZ.v b * SZ.v m)).
      on gpu_loc (y |-> sy') **
      pure (maxreduce_post (SZ.v b * SZ.v m) (SZ.v d) sx (chest1_to_seq sy'))
{
  maxreduce_dim_fw_f32_impl b m d x y #sx #sy
}
