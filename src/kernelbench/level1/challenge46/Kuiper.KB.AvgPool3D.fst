module Kuiper.KB.AvgPool3D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major, l1_forward }
open Kuiper.Monoid.Reduce.F32 { cmonoid_fadd_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce, windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
open Kuiper.Seq.Common { lseq_map }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module ML = FStar.Math.Lemmas
module SM = Kuiper.KB.ScalarMul

(* Verified, extractable 1-D pool output length, provably equal to the pure
   spec [pool_out_len_1d].  The C bridge calls this (per axis) instead of
   re-implementing the formula in unverified C. *)
let pool_out_len_1d_sz
  (l k s : szp) (p : sz) (d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)
  =
  let kspan  : sz = SZ.((d *^ (k -^ 1sz)) +^ 1sz) in
  let padded : sz = SZ.(l +^ (2sz *^ p)) in
  if SZ.(padded <^ kspan) then 0sz
  else SZ.(((padded -^ kspan) /^ s) +^ 1sz)

(* Verified, extractable reciprocal 1/k as f32 (extracts to
   1.0f / (float)(int64_t)(uint64_t)k); the per-axis average divisor is
   computed inside the verification boundary. *)
let avgpool_recip_f32 (k : szp) : f32 =
  div one (of_int (FStar.Int.Cast.uint64_to_int64
                     (FStar.SizeT.sizet_to_uint64 k)))

inline_for_extraction noextract
fn avgpool3d_axis_fw
  (#t : Type0) {| scalar t |}
  (m_inst : Kuiper.Monoid.Reduce.cmonoid t)
  (k s : szp)
  (p : sz)
  (d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l    : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
  (#lin  : layout2 bc l)     {| ctlayout lin  |}
  (#lout : layout2 bc l_out) {| ctlayout lout |}
  (input  : array2 t lin  { is_global input  })
  (output : array2 t lout { is_global output })
  (#fIn  : perm)
  (#sx   : chest2 t bc l)
  (#sout : chest2 t bc l_out)
  preserves
    cpu **
    on gpu_loc (input  |-> Frac fIn sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
  ensures
    on gpu_loc (output |->
      windowreduce_result m_inst sx
        k s p d l_out)
{
  windowreduce m_inst k s p d bc l l_out input output
}

inline_for_extraction noextract
let avgpool3d_axis_fw_f32 =
  fun k s p d bc l l_out #_ #_ #_ #_ input output #fIn #sx #sout ->
    avgpool3d_axis_fw #f32 cmonoid_fadd_f32 k s p d bc l l_out input output
      #fIn #sx #sout

let avgpool3d_axis_fw_rm_f32 =
  fun k s p d bc l l_out input output #fIn #sx #sout ->
    avgpool3d_axis_fw_f32 k s p d bc l l_out
      #(l2_row_major bc l)     #_
      #(l2_row_major bc l_out) #_
      input output
      #fIn #sx #sout

(* ── Self-allocating per-axis entry (mirrors #44 [avgpool1d_alloc]) ───── *)

(* Reshape glue: a [(m, cn)] row-major Array2 buffer viewed as a flat
   [m*cn] Array1 over the same store, and back. *)
ghost
fn reshape2to1
  (#et:Type) (#m #cn:nat)
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 et (l2_row_major m cn))
  (#s2 : chest2 et m cn)
  (#f : perm)
  requires
    a2 |-> Frac f s2
  ensures
    from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l2_row_major m cn) s2))
{
  tensor_concr a2;
  tensor_abs' (l1_forward p) (core a2)
}

ghost
fn reshape1to2
  (#et:Type) (#m #cn:nat)
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 et (l2_row_major m cn))
  (#s2 : chest2 et m cn)
  (#f : perm)
  requires
    from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l2_row_major m cn) s2))
  ensures
    a2 |-> Frac f s2
{
  tensor_concr (from_array (l1_forward p) (core a2));
  rewrite
    (core (from_array (l1_forward p) (core a2))
      |-> Frac f (to_seq (l1_forward p)
                    (from_seq (l1_forward p)
                       (to_seq (l2_row_major m cn) s2))))
  as
    (core a2 |-> Frac f (to_seq (l2_row_major m cn) s2));
  tensor_abs (l2_row_major m cn) (core a2) #f #s2;
  rewrite
    (from_array (l2_row_major m cn) (core a2) |-> Frac f s2)
  as
    (a2 |-> Frac f s2);
}

ghost
fn reshape1to2_eq
  (#et:Type) (#m #cn:nat)
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 et (l2_row_major m cn))
  (#s2 : chest2 et m cn)
  (#f : perm)
  (#e : chest1 et p)
  (#_ : squash (
     e == from_seq (l1_forward p)
            (to_seq (l2_row_major m cn) s2)))
  requires
    from_array (l1_forward p) (core a2) |-> Frac f e
  ensures
    a2 |-> Frac f s2
{
  rewrite
    (from_array (l1_forward p) (core a2) |-> Frac f e)
  as
    (from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l2_row_major m cn) s2)));
  reshape1to2 p a2 #s2 #f;
}

(* Mapping [mul c] over the flattened row-major sequence equals flattening the
   scaled matrix (pure index reasoning; both sides are length-[p] sequences). *)
#push-options ""
let smul_reshape_eq2
  (c : f32) (#m #cn : nat) (p:nat) (_:squash (p == m * cn))
  (s2 : chest2 f32 m cn)
  : Lemma
    (chest_map (mul c)
        (from_seq (l1_forward p) (to_seq (l2_row_major m cn) s2))
     == from_seq (l1_forward p)
           (to_seq (l2_row_major m cn)
              (mk2 (fun (i:natlt m) (j:natlt cn) -> mul c (acc2 s2 i j)))))
  = let l2 = l2_row_major m cn in
    let l1 = l1_forward p in
    let scaled = mk2 (fun (i:natlt m) (j:natlt cn) -> mul c (acc2 s2 i j)) in
    let lhs = chest_map (mul c) (from_seq l1 (to_seq l2 s2)) in
    let rhs = from_seq l1 (to_seq l2 scaled) in
    let aux (i:abs (p @| INil)) : Lemma (acc lhs i == acc rhs i) =
      let q = l1.imap.f i in
      ()
    in
    Classical.forall_intro aux;
    Kuiper.Chest.lemma_equal_intro lhs rhs;
    Kuiper.Chest.ext lhs rhs
#pop-options

(* Congruence: scaling-matrices built from equal scalars and equal source
   matrices are equal. *)
let scale_matrix_cong (#m #cn:nat) (c1 c2 : f32) (e1 e2 : chest2 f32 m cn)
  : Lemma (requires c1 == c2 /\ e1 == e2)
          (ensures mk2 (fun (i:natlt m) (j:natlt cn) -> mul c1 (acc2 e1 i j))
                == mk2 (fun (i:natlt m) (j:natlt cn) -> mul c2 (acc2 e2 i j)))
  = ()

inline_for_extraction noextract
fn avgpool3d_axis_alloc
  (k s : szp)
  (p : sz)
  (d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l : szp { SZ.fits (SZ.v bc * SZ.v l) })
  (input : array2 f32 (l2_row_major bc l) { is_global input })
  (#fIn : perm)
  (#sx  : chest2 f32 bc l)
  preserves
    cpu **
    on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
    pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
    pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
    pure (SZ.fits (pool_out_len_1d l k s p d
                     * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.fits (SZ.v bc *
            pool_out_len_1d l k s p d)) **
    pure (SZ.v bc *
            pool_out_len_1d l k s p d
          <= max_blocks * max_threads)
  returns r : (lo:sz { SZ.v lo == pool_out_len_1d l k s p d }
               & array2 f32 (l2_row_major bc lo))
  ensures
    on gpu_loc ((dsnd r) |->
      mk2 (fun (i:natlt bc) (j:natlt (dfst r)) ->
        mul (avgpool_recip_f32 k)
            (acc2 (windowreduce_result cmonoid_fadd_f32 sx
                       k s p d (dfst r)) i j))) **
    pure (SZ.v (dfst r) ==
            pool_out_len_1d l k s p d)
{
  let l_out = pool_out_len_1d_sz l k s p d;
  let output = alloc0 #f32 (bc *^ l_out) (l2_row_major bc l_out);
  avgpool3d_axis_fw_rm_f32 k s p d bc l l_out input output;
  (* output |-> wr, where wr is the per-window SUM. *)
  let inv_k = avgpool_recip_f32 k;
  let n : szp = bc *^ l_out;
  assert pure (SZ.v n == SZ.v bc * SZ.v l_out);
  let pp : erased nat = SZ.v n;
  let wr : chest2 f32 bc l_out =
    hide (windowreduce_result cmonoid_fadd_f32 sx
            k s p d l_out);
  (* View the row-major output buffer as a flat array1 over the same store. *)
  map_loc gpu_loc (fun () -> reshape2to1 pp output);
  (* Verified in-place /K scale on the flat view. *)
  SM.smul_fw_f32 inv_k n (from_array (l1_forward pp) (core output));
  (* Reflect the flat scale back to the matrix view. *)
  smul_reshape_eq2 inv_k #bc #l_out pp () (reveal wr);
  map_loc gpu_loc (fun () ->
    reshape1to2_eq pp output
      #(mk2 (fun (i:natlt bc) (j:natlt l_out) ->
          mul inv_k (acc2 (reveal wr) i j)))
      #_
      #(chest_map (mul inv_k)
          (from_seq (l1_forward pp)
             (to_seq (l2_row_major bc l_out) (reveal wr)))));
  scale_matrix_cong #bc #l_out
    inv_k (avgpool_recip_f32 k)
    (reveal wr)
    (windowreduce_result cmonoid_fadd_f32 sx
       k s p d l_out);
  rewrite
    (on gpu_loc (output |->
       mk2 (fun (i:natlt bc) (j:natlt l_out) ->
         mul inv_k (acc2 (reveal wr) i j))))
  as
    (on gpu_loc (output |->
       mk2 (fun (i:natlt bc) (j:natlt l_out) ->
         mul (avgpool_recip_f32 k)
             (acc2 (windowreduce_result cmonoid_fadd_f32 sx
                        k s p d l_out) i j))));
  (| (l_out <: (lo:sz { SZ.v lo == pool_out_len_1d l k s p d })), output |)
}

let avgpool3d_axis_alloc_f32 =
  fun k s p d bc l input #fIn #sx ->
    avgpool3d_axis_alloc k s p d bc l input #fIn #sx
