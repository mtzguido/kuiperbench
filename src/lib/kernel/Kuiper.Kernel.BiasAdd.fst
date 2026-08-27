module Kuiper.Kernel.BiasAdd

(* Implementation of [Kuiper.Kernel.BiasAdd].

   One thread per output element.  Thread [tid] decodes [(i, j) =
   (tid / n, tid % n)], reads [C[i,j]] (via [tensor_read], which yields
   [acc2]) and [bias[j]], and writes [add C[i,j] bias[j]] into [y[tid]].

   The read of the second array [bias] by the computed sub-index [j =
   tid % n] mirrors the [gbias] read of [Kuiper.Kernel.Conv2D.Naive].
   Setup/teardown/sendability are discharged at the [kdesc] level. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
open Kuiper.Bijection { ( =~ ) }
open Kuiper.Seq.Common { op_At_Bang }
module Seq = FStar.Seq
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

(* Bijection between the abstract 1-D tensor index [(k, ())] and a plain
   [natlt len], used to (un)reindex a forevery over rank-1 tensor cells. *)
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }


(* [a < mm * n /\ n > 0  ==>  a / n < mm]. *)
let div_bound (a : nat) (mm : nat) (n : pos)
  : Lemma (requires a < mm * n) (ensures a / n < mm)
  = Math.division_propriety a n;
    Math.multiplication_order_lemma (a / n) mm n

let bias_add_at_ij
  (#t:Type0) {| scalar t |}
  (m n : nat)
  (eC : EM.chest2 t m n)
  (sbias : chest1 t n)
  (i : natlt m) (j : natlt n)
  : Lemma (bias_add_at m n eC sbias (i * n + j) == add (acc2 eC i j) (acc1 sbias j))
  = Math.lemma_mod_plus j i n;       (* (i*n + j) % n == j % n == j *)
    Math.lemma_div_plus j i n;       (* (i*n + j) / n == i + j/n == i *)
    Math.small_div j n;
    Math.small_mod j n

(* Per-thread pre/post predicates. *)

unfold
let kpre
  (#t:Type0) {| scalar t |}
  (m n : szp)
  (#lC : layout2 m n)
  (#lbias : layout1 n)
  (#ly : layout1 (m * n))
  (gC : array2 t lC)
  (gbias : array1 t lbias)
  (gy : array1 t ly)
  (eC : EM.chest2 t m n)
  (sbias : chest1 t n)
  (sy0 : chest1 t (m * n))
  (fc fb : perm)
  (tid : natlt (m * n))
  : slprop
  = (gC |-> Frac (fc /. (m * n)) eC) **
    (gbias |-> Frac (fb /. (m * n)) sbias) **
    (Cell gy (idx1 tid) |-> acc1 sy0 tid)

unfold
let kpost
  (#t:Type0) {| scalar t |}
  (m n : szp)
  (#lC : layout2 m n)
  (#lbias : layout1 n)
  (#ly : layout1 (m * n))
  (gC : array2 t lC)
  (gbias : array1 t lbias)
  (gy : array1 t ly)
  (eC : EM.chest2 t m n)
  (sbias : chest1 t n)
  (fc fb : perm)
  (tid : natlt (m * n))
  : slprop
  = (gC |-> Frac (fc /. (m * n)) eC) **
    (gbias |-> Frac (fb /. (m * n)) sbias) **
    (Cell gy (idx1 tid) |-> bias_add_at m n eC sbias tid)

#push-options "--z3rlimit 200 --fuel 2 --ifuel 1"

(* Per-thread body: decode [tid], read [C[i,j]] and [bias[j]], write the sum. *)
inline_for_extraction noextract
fn kf
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lC : layout2 m n) {| ctlayout lC |}
  (#lbias : layout1 n) {| ctlayout lbias |}
  (#ly : layout1 (m * n)) {| ctlayout ly |}
  (gC : array2 t lC)
  (gbias : array1 t lbias)
  (gy : array1 t ly)
  (#eC : EM.chest2 t m n)
  (#sbias : chest1 t n)
  (#sy0 : chest1 t (m * n))
  (#fc #fb : perm)
  (#_ : squash (SZ.fits (m * n)))
  (tid : szlt (m * n))
  ()
  norewrite
  requires
    gpu **
    kpre #t m n #lC #lbias #ly gC gbias gy eC sbias sy0 fc fb tid
  ensures
    gpu **
    kpost #t m n #lC #lbias #ly gC gbias gy eC sbias fc fb tid
{
  let i : szlt m = tid /^ n;
  let j : szlt n = tid %^ n;
  let cv = tensor_read gC ((i <: szlt m), ((j <: szlt n), ()));
  let bv = tensor_read gbias ((j <: szlt n), ());
  tensor_write_cell gy ((tid <: szlt (m * n)), ()) (add cv bv);
}

#pop-options

#push-options "--z3rlimit 60"

(* Ghost setup: factor the launcher frame into [m*n] per-thread slices. *)
ghost
fn bias_add_setup
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (nthr : szp { SZ.v nthr == m * n })
  (#lC : layout2 m n)
  (#lbias : layout1 n)
  (#ly : layout1 (m * n))
  (gC : array2 t lC)
  (gbias : array1 t lbias)
  (gy : array1 t ly)
  (#eC : EM.chest2 t m n)
  (#sbias : chest1 t n)
  (#sy0 : chest1 t (m * n))
  (#fc #fb : perm)
  (#_ : squash (SZ.fits (m * n)))
  ()
  norewrite
  requires
    (gC |-> Frac fc eC) **
    (gbias |-> Frac fb sbias) **
    (gy |-> sy0)
  ensures
    (forall+ (tid : natlt nthr).
       kpre #t m n #lC #lbias #ly gC gbias gy eC sbias sy0 fc fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
{
  tensor_pts_to_ref gy;
  tensor_share_n gC (m * n);
  tensor_share_n gbias (m * n);
  tensor_explode gy;
  forevery_iso (abs_bij #(m * n))
    (fun (i : abs ((m * n) @| INil)) -> Cell gy i |-> acc sy0 i);
  forevery_ext
    (fun (y : natlt (m * n)) -> Cell gy (abs_bij.gg y) |-> acc sy0 (abs_bij.gg y))
    (fun (i : natlt (m * n)) -> Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (m * n)) ->
       gbias |-> Frac (fb /. (m * n)) sbias)
    (fun (i : natlt (m * n)) ->
       Cell gy (idx1 i) |-> acc1 sy0 i);
  forevery_zip
    (fun (_ : natlt (m * n)) ->
       gC |-> Frac (fc /. (m * n)) eC)
    (fun (i : natlt (m * n)) ->
       (gbias |-> Frac (fb /. (m * n)) sbias) **
       (Cell gy (idx1 i) |-> acc1 sy0 i));
  forevery_rw_size (m * n) nthr;
  ()
}

#pop-options

#push-options "--z3rlimit 60"

(* Ghost teardown: gather the per-thread slices into the launcher post. *)
ghost
fn bias_add_teardown
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (nthr : szp { SZ.v nthr == m * n })
  (#lC : layout2 m n)
  (#lbias : layout1 n)
  (#ly : layout1 (m * n))
  (gC : array2 t lC)
  (gbias : array1 t lbias)
  (gy : array1 t ly)
  (#eC : EM.chest2 t m n)
  (#sbias : chest1 t n)
  (#fc #fb : perm)
  (#_ : squash (SZ.fits (m * n)))
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
       kpost #t m n #lC #lbias #ly gC gbias gy eC sbias fc fb tid) **
    pure (SZ.fits (tlayout_ulen ly))
  ensures
    (gC |-> Frac fc eC) **
    (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 t (m * n)).
       (gy |-> sy) **
       pure (forall (tid : nat{tid < m * n}).
               acc1 sy tid == bias_add_at m n eC sbias tid))
{
  forevery_rw_size nthr (m * n)
    #(kpost #t m n #lC #lbias #ly gC gbias gy eC sbias fc fb);
  forevery_unzip
    (fun (_ : natlt (m * n)) ->
       gC |-> Frac (fc /. (m * n)) eC)
    (fun (i : natlt (m * n)) ->
       (gbias |-> Frac (fb /. (m * n)) sbias) **
       (Cell gy (idx1 i) |-> bias_add_at m n eC sbias i));
  forevery_unzip
    (fun (_ : natlt (m * n)) ->
       gbias |-> Frac (fb /. (m * n)) sbias)
    (fun (i : natlt (m * n)) ->
       Cell gy (idx1 i) |-> bias_add_at m n eC sbias i);
  tensor_gather_n gC (m * n);
  tensor_gather_n gbias (m * n);
  let sy : chest1 t (m * n) =
    hide (mk1 #t #(m * n)
            (fun (tid : natlt (m * n)) -> bias_add_at m n eC sbias tid));
  forevery_ext
    (fun (i : natlt (m * n)) ->
       Cell gy (idx1 i) |-> bias_add_at m n eC sbias i)
    (fun (i : natlt (m * n)) ->
       Cell gy (idx1 i) |-> acc1 (reveal sy) i);
  forevery_ext
    (fun (i : natlt (m * n)) ->
       Cell gy (idx1 i) |-> acc1 (reveal sy) i)
    (fun (y : natlt (m * n)) ->
       Cell gy (abs_bij.gg y) |-> acc (reveal sy) (abs_bij.gg y));
  forevery_iso_back (abs_bij #(m * n))
    (fun (i : abs ((m * n) @| INil)) -> Cell gy i |-> acc (reveal sy) i);
  tensor_implode gy;
  assert pure (forall (tid : nat{tid < m * n}).
                 acc1 (reveal sy) tid == bias_add_at m n eC sbias tid);
  ()
}

#pop-options

#push-options "--z3rlimit 100 --fuel 2 --ifuel 1"

inline_for_extraction noextract
let kdesc
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lC : layout2 m n) {| ctlayout lC |}
  (#lbias : layout1 n) {| ctlayout lbias |}
  (#ly : layout1 (m * n)) {| ctlayout ly |}
  (gC : array2 t lC)
  (gbias : array1 t lbias)
  (gy : array1 t ly)
  (#eC : EM.chest2 t m n)
  (#sbias : chest1 t n)
  (#sy0 : chest1 t (m * n))
  (#fc #fb : perm)
  (#_ : squash (is_global gC /\ is_global gbias /\
                is_global gy /\
                bias_add_size_req m n))
  : kernel_desc
      ((gC |-> Frac fc eC) **
       (gbias |-> Frac fb sbias) **
       (gy |-> sy0))
      ((gC |-> Frac fc eC) **
       (gbias |-> Frac fb sbias) **
       (exists* (sy : chest1 t (m * n)).
          (gy |-> sy) **
          pure (forall (tid : nat{tid < m * n}).
                  acc1 sy tid == bias_add_at m n eC sbias tid)))
  = [@@inline_let] let nthr : (x : szp { SZ.v x == m * n }) = m *^ n in {
  nthr = nthr;
  frame = pure (SZ.fits (tlayout_ulen ly));
  setup    = bias_add_setup m n nthr gC gbias gy;
  teardown = bias_add_teardown m n nthr gC gbias gy;
  kpre  = kpre #t m n #lC #lbias #ly gC gbias gy eC sbias sy0 fc fb;
  kpost = kpost #t m n #lC #lbias #ly gC gbias gy eC sbias fc fb;
  f = kf m n gC gbias gy;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _

#pop-options

inline_for_extraction noextract
fn bias_add_gpu
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lC : layout2 m n) {| ctlayout lC |}
  (#lbias : layout1 n) {| ctlayout lbias |}
  (#ly : layout1 (SZ.v m * SZ.v n)) {| ctlayout ly |}
  (gC : array2 t lC)
  (gbias : array1 t lbias)
  (gy : array1 t ly)
  (#eC : EM.chest2 t m n)
  (#sbias : chest1 t n)
  (#sy0 : chest1 t (SZ.v m * SZ.v n))
  (#fc #fb : perm)
  preserves cpu
  requires
    on gpu_loc (gC |-> Frac fc eC) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (is_global gC /\ is_global gbias /\ is_global gy /\
          bias_add_size_req m n)
  ensures
    on gpu_loc (gC |-> Frac fc eC) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 t (SZ.v m * SZ.v n)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < SZ.v m * SZ.v n}).
               acc1 sy tid == bias_add_at m n eC sbias tid))
{
  launch_sync (kdesc m n gC gbias gy)
}

fn bias_add_f32
  (m n : szp)
  (gC : array2 f32 (l2_row_major m n) { is_global gC })
  (gbias : array1 f32 (l1_forward n) { is_global gbias })
  (gy : array1 f32 (l1_forward (SZ.v m * SZ.v n)) { is_global gy })
  (#eC : EM.chest2 f32 m n)
  (#sbias : chest1 f32 n)
  (#sy0 : chest1 f32 (SZ.v m * SZ.v n))
  (#fc #fb : perm)
  preserves cpu
  requires
    on gpu_loc (gC |-> Frac fc eC) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0) **
    pure (bias_add_size_req m n)
  ensures
    on gpu_loc (gC |-> Frac fc eC) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 f32 (SZ.v m * SZ.v n)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < SZ.v m * SZ.v n}).
               acc1 sy tid == bias_add_at m n eC sbias tid))
{
  bias_add_gpu m n gC gbias gy
}
