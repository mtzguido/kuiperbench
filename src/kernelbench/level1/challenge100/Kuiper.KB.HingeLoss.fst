module Kuiper.KB.HingeLoss

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor.Layout { from_seq, to_seq }
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, l2_col_major }
open Kuiper.Spec.HingeLoss
module SZ = Kuiper.SizeT
module HRed = Kuiper.Kernel.HReduce
module Map = Kuiper.Kernel.Map
module RB = Kuiper.Kernel.RowBroadcast
module GT = Kuiper.Ghost.TensorTranspose
module EM = Kuiper.EMatrix
module Seq = FStar.Seq

(* Keep the extracted pointwise operation monomorphic. *)
inline_for_extraction noextract
let hinge_step_f32 (prediction target : f32) : f32 =
  fmax (zero #f32) (sub (one #f32) (mul prediction target))

(* [row_broadcast] passes the broadcast value first and the matrix cell
   second. *)
inline_for_extraction noextract
let hinge_broadcast_step_f32 (target prediction : f32) : f32 =
  hinge_step_f32 prediction target

let hinge_step_approx
  (prediction target : f32)
  (rprediction rtarget : real)
  : Lemma
      (requires prediction %~ rprediction /\ target %~ rtarget)
      (ensures hinge_step_f32 prediction target
               %~ real_hinge_step rprediction rtarget)
  = to_real_ok (zero #f32);
    to_real_ok (one #f32);
    a_mul prediction target rprediction rtarget;
    sub_approx (one #f32) (mul prediction target)
      1.0R (rprediction *. rtarget);
    fmax_approx (zero #f32) (sub (one #f32) (mul prediction target))
      0.0R (1.0R -. rprediction *. rtarget)

(* Transposing only the logical view turns the row-major (B,N) scratch into
   a column-major (N,B) matrix.  Broadcasting targets over its rows therefore
   updates physical cell [i*N+j] with [targets[j]]. *)
#push-options "--z3rlimit 60"
let hinge_broadcast_approx
  (#b #n : pos)
  (sp : chest2 f32 b n)
  (st : chest1 f32 n)
  (rp : chest2 real b n)
  (rt : chest1 real n)
  : Lemma
      (requires sp %~ rp /\ st %~ rt)
      (ensures
        RB.s_row_broadcast hinge_broadcast_step_f32 st (EM.mtranspose sp)
          %~ real_hinge_matrix b n rp rt)
  = let lhs =
      RB.s_row_broadcast hinge_broadcast_step_f32 st (EM.mtranspose sp) in
    let rhs = real_hinge_matrix b n rp rt in
    let aux (idx : Kuiper.Shape.abs (n @| b @| INil))
      : Lemma (acc lhs idx %~ acc rhs idx) =
      let (j, (i, ())) = idx in
      hinge_step_approx (acc2 sp i j) (acc1 st j)
        (acc2 rp i j) (acc1 rt j)
    in
    Classical.forall_intro aux
#pop-options

(* Rank-only view used before the final reduction. *)
ghost
fn reshape_col2to1
  (#et : Type) (#m #n : nat)
  (p : nat) (#_ : squash (p == m * n))
  (a2 : array2 et (l2_col_major m n))
  (#s2 : chest2 et m n)
  (#f : perm)
  requires a2 |-> Frac f s2
  ensures
    from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                    (to_seq (l2_col_major m n) s2))
{
  tensor_concr a2;
  tensor_abs' (l1_forward p) (core a2)
}

(* Copy predictions into a private row-major scratch buffer, preserving the
   public input.  This is an ordinary device copy, not semantic preprocessing. *)
inline_for_extraction noextract
fn copy_row_major_f32
  (b n : szp)
  (src : array2 f32 (l2_row_major b n) { is_global src })
  (dst : array2 f32 (l2_row_major b n) { is_global dst })
  (#ss #sd : chest2 f32 b n)
  (#f : perm)
  preserves cpu ** on gpu_loc (src |-> Frac f ss)
  requires on gpu_loc (dst |-> sd)
  ensures on gpu_loc (dst |-> ss)
{
  let elems : szp = b *^ n;
  map_loc gpu_loc
    #(dst |-> sd)
    #(core dst |-> to_seq (l2_row_major b n) sd)
    fn _ { tensor_concr dst; };
  map_loc gpu_loc
    #(src |-> Frac f ss)
    #(core src |-> Frac f (to_seq (l2_row_major b n) ss))
    fn _ { tensor_concr src; };
  gpu_memcpy_device_to_device (core dst) (core src) elems;
  map_loc gpu_loc
    #(core src |-> Frac f (to_seq (l2_row_major b n) ss))
    #(src |-> Frac f ss)
    fn _ {
      tensor_abs (l2_row_major b n) (core src);
      rewrite (from_array (l2_row_major b n) (core src) |-> Frac f ss)
           as (src |-> Frac f ss);
    };
  map_loc gpu_loc
    #(core dst |-> to_seq (l2_row_major b n) ss)
    #(dst |-> ss)
    fn _ {
      tensor_abs (l2_row_major b n) (core dst);
      rewrite (from_array (l2_row_major b n) (core dst) |-> ss)
           as (dst |-> ss);
    }
}

(* Flattening a pair of pointwise-related column-major matrices preserves the
   relation at every physical element. *)
#push-options "--z3rlimit 60"
let flatten_col_approx
  (#m #n : nat)
  (sf : chest2 f32 m n)
  (sr : chest2 real m n)
  : Lemma
      (requires sf %~ sr)
      (ensures
        from_seq (l1_forward (m * n)) (to_seq (l2_col_major m n) sf)
          %~ from_seq (l1_forward (m * n))
                (to_seq (l2_col_major m n) sr))
  = let lhs =
      from_seq (l1_forward (m * n)) (to_seq (l2_col_major m n) sf) in
    let rhs =
      from_seq (l1_forward (m * n)) (to_seq (l2_col_major m n) sr) in
    let aux (i : natlt (m * n))
      : Lemma (acc1 lhs i %~ acc1 rhs i) = () in
    Classical.forall_intro aux
#pop-options

let chest1_seq_roundtrip (#et : Type) (#n : nat) (s : lseq et n)
  : Lemma (Seq.equal (chest1_to_seq (seq_to_chest1 s)) s)
  = ()

let l1_from_seq_eq (#et : Type) (#n : nat) (s : lseq et n)
  : Lemma (equal (from_seq (l1_forward n) s) (seq_to_chest1 s))
  = ()

let chest_map_id (#et : Type) (#n : nat) (s : chest1 et n)
  : Lemma (equal (chest_map id s) s)
  = ()

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn hinge_loss_broadcast
  (b : szp)
  (n : szp {
     SZ.v b * SZ.v n <= max_blocks * max_threads /\
     SZ.fits (SZ.v b * SZ.v n) /\
     SZ.fits (SZ.v b * SZ.v n + max_threads) })
  (predictions : array2 f32 (l2_row_major b n) { is_global predictions })
  (targets     : array1 f32 (l1_forward n) { is_global targets })
  (#sp : chest2 f32 b n)
  (#st : chest1 f32 n)
  (rp : erased (chest2 real b n))
  (rt : erased (chest1 real n))
  (#fp #ft : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (predictions |-> Frac fp sp) **
    on gpu_loc (targets |-> Frac ft st) **
    pure (sp %~ rp /\ st %~ rt)
  returns out : array1 f32 (l1_forward 1)
  ensures
    (exists* (sout : chest1 f32 1).
       on gpu_loc (out |-> sout) **
       pure (acc1 sout 0 %~ real_hinge_broadcast b n rp rt))
{
  let elems : szp = b *^ n;

  (* Preserve predictions by doing all pointwise work in private scratch. *)
  let scratch = alloc0 #f32 elems (l2_row_major b n);
  copy_row_major_f32 b n predictions scratch;

  (* Same storage, transposed logical view: rows are target indices. *)
  map_loc gpu_loc (fun () -> GT.ghost_transpose1 scratch);
  RB.row_broadcast hinge_broadcast_step_f32 n b targets (GT.row2col scratch);

  with sbroadcast. assert
    (on gpu_loc (GT.row2col scratch |-> reveal sbroadcast));
  hinge_broadcast_approx (reveal sp) (reveal st) (reveal rp) (reveal rt);
  assert pure
    (reveal sbroadcast %~ real_hinge_matrix b n (reveal rp) (reveal rt));

  (* The column-major (N,B) physical order is the original row-major (B,N)
     order, so a flat reduction visits exactly all canonical output terms. *)
  map_loc gpu_loc (fun () ->
    reshape_col2to1 (SZ.v elems) (GT.row2col scratch));

  let vr : chest1 real elems =
    hide (from_seq (l1_forward elems)
      (real_hinge_terms b n (reveal rp) (reveal rt)));
  flatten_col_approx (reveal sbroadcast)
    (real_hinge_matrix b n (reveal rp) (reveal rt));
  assert pure
    (from_seq (l1_forward elems)
       (to_seq (l2_col_major n b) (reveal sbroadcast)) %~ reveal vr);

  assert pure (SZ.fits (SZ.v b * SZ.v n));
  let sum = HRed.reduce #f32 id id 1024sz elems
    (from_array (l1_forward elems) (core (GT.row2col scratch)))
    #(from_seq (l1_forward elems)
        (to_seq (l2_col_major n b) (reveal sbroadcast)))
    vr;
  chest_map_id (reveal vr);
  l1_from_seq_eq
    (real_hinge_terms b n (reveal rp) (reveal rt));
  chest1_seq_roundtrip
    (real_hinge_terms b n (reveal rp) (reveal rt));
  assert pure
    (sum %~ rsum (real_hinge_terms b n (reveal rp) (reveal rt)));
  free (from_array (l1_forward elems) (core (GT.row2col scratch)));

  let total64 : Int64.t = FStar.Int.Cast.uint64_to_int64
    (FStar.SizeT.sizet_to_uint64 elems);
  assert pure (Int64.v total64 == SZ.v b * SZ.v n);
  let denom : f32 = of_int total64;
  of_int_approx #f32 total64;
  assert pure (denom %~ Real.of_int (b * n));
  let mean : f32 = div sum denom;
  div_approx sum denom
    (rsum (real_hinge_terms b n (reveal rp) (reveal rt)))
    (Real.of_int (b * n));
  assert pure
    (mean %~ real_hinge_broadcast b n (reveal rp) (reveal rt));

  (* Ownership of this one-element GPU allocation is transferred to C++. *)
  let out = alloc0 #f32 1sz (l1_forward 1);
  Map.map_gpu (fun _ -> mean) 1sz out;
  with sout. assert (on gpu_loc (out |-> sout));
  assert pure (acc1 sout 0 == mean);
  out
}
#pop-options

let hinge_loss_broadcast_f32 = hinge_loss_broadcast
