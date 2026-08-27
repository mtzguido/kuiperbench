module Kuiper.Kernel.RowBroadcast2

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
open Kuiper.EMatrix
open Kuiper.Seq.Common
open Kuiper.Tensor

unfold
let tid_to_cell (m n : nat) (tid : natlt (m * n))
  : abs (m @| n @| INil) =
  idx2 (tid / n) (tid % n)

unfold
let kpre
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (m n : szp)
  (#la1 : layout1 m) {| ctlayout la1 |}
  (a1 : array1 t la1)
  (#la2 : layout1 m) {| ctlayout la2 |}
  (a2 : array1 t la2)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#fA1 : perm)
  (#fA2 : perm)
  (#sa1 : chest1 t m)
  (#sa2 : chest1 t m)
  (#sb : chest2 t m n)
  (tid : natlt (m * n))
  : slprop
  = a1 |-> Frac (fA1 /. (m * n)) sa1 **
    a2 |-> Frac (fA2 /. (m * n)) sa2 **
    Cell b (tid_to_cell m n tid) |-> acc2 sb (tid / n) (tid % n)

unfold
let kpost
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (m n : szp)
  (#la1 : layout1 m) {| ctlayout la1 |}
  (a1 : array1 t la1)
  (#la2 : layout1 m) {| ctlayout la2 |}
  (a2 : array1 t la2)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#fA1 : perm)
  (#fA2 : perm)
  (#sa1 : chest1 t m)
  (#sa2 : chest1 t m)
  (#sb : chest2 t m n)
  (tid : natlt (m * n))
  : slprop
  = a1 |-> Frac (fA1 /. (m * n)) sa1 **
    a2 |-> Frac (fA2 /. (m * n)) sa2 **
    Cell b (tid_to_cell m n tid)
      |-> acc2 (s_row_broadcast2 f sa1 sa2 sb) (tid / n) (tid % n)

ghost
fn setup
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (m n : szp)
  (nthr : szp { SZ.v nthr == m * n })
  (#la1 : layout1 m) {| ctlayout la1 |}
  (a1 : array1 t la1)
  (#la2 : layout1 m) {| ctlayout la2 |}
  (a2 : array1 t la2)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#fA1 : perm)
  (#fA2 : perm)
  (#sa1 : chest1 t m)
  (#sa2 : chest1 t m)
  (#sb : chest2 t m n)
  ()
  norewrite
  requires
    a1 |-> Frac fA1 sa1 **
    a2 |-> Frac fA2 sa2 **
    b |-> sb
  ensures
    (forall+ (tid : natlt nthr).
      kpre #t f m n #la1 a1 #la2 a2 #lb b #fA1 #fA2 #sa1 #sa2 #sb tid) **
    pure (SZ.fits (tlayout_ulen lb))
{
  tensor_share_n a1 (m * n);
  tensor_share_n a2 (m * n);
  tensor_ilower2 b;
  forevery_unfactor' (m * n) m n (fun r c ->
    Cell b (idx2 r c) |-> acc2 sb r c);
  forevery_zip #(natlt (m * n))
    (fun _ -> a2 |-> Frac (fA2 /. (m * n)) sa2)
    (fun (tid : natlt (m * n)) ->
       Cell b (idx2 ((tid / n) <: natlt m) ((tid % n) <: natlt n))
         |-> acc2 sb (tid / n) (tid % n));
  forevery_zip #(natlt (m * n))
    (fun _ -> a1 |-> Frac (fA1 /. (m * n)) sa1)
    (fun (tid : natlt (m * n)) ->
       (a2 |-> Frac (fA2 /. (m * n)) sa2) **
       (Cell b (idx2 ((tid / n) <: natlt m) ((tid % n) <: natlt n))
          |-> acc2 sb (tid / n) (tid % n)));
  forevery_ext #(natlt (m * n))
    (fun (tid : natlt (m * n)) ->
       (a1 |-> Frac (fA1 /. (m * n)) sa1) **
       ((a2 |-> Frac (fA2 /. (m * n)) sa2) **
        (Cell b (idx2 ((tid / n) <: natlt m) ((tid % n) <: natlt n))
           |-> acc2 sb (tid / n) (tid % n))))
    (fun tid -> kpre #t f m n #la1 a1 #la2 a2 #lb b #fA1 #fA2 #sa1 #sa2 #sb tid);
  forevery_rw_size (m * n) nthr;
  ()
}

ghost
fn teardown
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (m n : szp)
  (nthr : szp { SZ.v nthr == m * n })
  (#la1 : layout1 m) {| ctlayout la1 |}
  (a1 : array1 t la1)
  (#la2 : layout1 m) {| ctlayout la2 |}
  (a2 : array1 t la2)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#fA1 : perm)
  (#fA2 : perm)
  (#sa1 : chest1 t m)
  (#sa2 : chest1 t m)
  (#sb : chest2 t m n)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost #t f m n #la1 a1 #la2 a2 #lb b #fA1 #fA2 #sa1 #sa2 #sb tid) **
    pure (SZ.fits (tlayout_ulen lb))
  ensures
    a1 |-> Frac fA1 sa1 **
    a2 |-> Frac fA2 sa2 **
    b |-> s_row_broadcast2 f sa1 sa2 sb
{
  forevery_rw_size nthr (m * n);
  forevery_ext #(natlt (m * n))
    (fun tid -> kpost #t f m n #la1 a1 #la2 a2 #lb b #fA1 #fA2 #sa1 #sa2 #sb tid)
    (fun (tid : natlt (m * n)) ->
       (a1 |-> Frac (fA1 /. (m * n)) sa1) **
       ((a2 |-> Frac (fA2 /. (m * n)) sa2) **
        (Cell b (idx2 ((tid / n) <: natlt m) ((tid % n) <: natlt n))
           |-> acc2 (s_row_broadcast2 f sa1 sa2 sb) (tid / n) (tid % n))));
  forevery_unzip #(natlt (m * n))
    (fun _ -> a1 |-> Frac (fA1 /. (m * n)) sa1)
    (fun (tid : natlt (m * n)) ->
       (a2 |-> Frac (fA2 /. (m * n)) sa2) **
       (Cell b (idx2 ((tid / n) <: natlt m) ((tid % n) <: natlt n))
          |-> acc2 (s_row_broadcast2 f sa1 sa2 sb) (tid / n) (tid % n)));
  tensor_gather_n a1 (m * n);
  forevery_unzip #(natlt (m * n))
    (fun _ -> a2 |-> Frac (fA2 /. (m * n)) sa2)
    (fun (tid : natlt (m * n)) ->
       Cell b (idx2 ((tid / n) <: natlt m) ((tid % n) <: natlt n))
         |-> acc2 (s_row_broadcast2 f sa1 sa2 sb) (tid / n) (tid % n));
  tensor_gather_n a2 (m * n);
  forevery_factor' (m * n) m n (fun r c ->
    Cell b (idx2 r c) |-> acc2 (s_row_broadcast2 f sa1 sa2 sb) r c);
  tensor_iraise2 b;
  ()
}

inline_for_extraction noextract
fn kf
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (m n : szp)
  (#la1 : layout1 m) {| ctlayout la1 |}
  (a1 : array1 t la1)
  (#la2 : layout1 m) {| ctlayout la2 |}
  (a2 : array1 t la2)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#fA1 : perm)
  (#fA2 : perm)
  (#sa1 : chest1 t m)
  (#sa2 : chest1 t m)
  (#sb : chest2 t m n)
  (tid : szlt (m * n))
  ()
  preserves gpu
  requires
    kpre #t f m n #la1 a1 #la2 a2 #lb b #fA1 #fA2 #sa1 #sa2 #sb tid
  ensures
    kpost #t f m n #la1 a1 #la2 a2 #lb b #fA1 #fA2 #sa1 #sa2 #sb tid
{
  let row : sz = tid /^ n; assert rewrites_to row (tid /^ n);
  let col : sz = tid %^ n; assert rewrites_to col (tid %^ n);
  let x = tensor_read_cell b ((row <: szlt m), ((col <: szlt n), ()));
  let v1 = tensor_read a1 ((row <: szlt m), ());
  let v2 = tensor_read a2 ((row <: szlt m), ());
  tensor_write_cell b ((row <: szlt m), ((col <: szlt n), ())) (f x v1 v2);
}

inline_for_extraction noextract
let kdesc
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la1 : layout1 m) {| ctlayout la1 |}
  (a1 : array1 t la1)
  (#la2 : layout1 m) {| ctlayout la2 |}
  (a2 : array1 t la2)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#_ : squash (is_global a1))
  (#_ : squash (is_global a2))
  (#_ : squash (is_global b))
  (#fA1 : perm)
  (#fA2 : perm)
  (#sa1 : chest1 t m)
  (#sa2 : chest1 t m)
  (#sb : chest2 t m n)
  : kernel_desc (requires a1 |-> Frac fA1 sa1 ** a2 |-> Frac fA2 sa2 ** b |-> sb)
                (ensures  a1 |-> Frac fA1 sa1 ** a2 |-> Frac fA2 sa2 ** b |-> s_row_broadcast2 f sa1 sa2 sb)
  = [@@inline_let] let nthr : (x : szp { SZ.v x == m * n }) = m *^ n in {
    nthr = nthr;
    f = (fun (tid : szlt nthr) ->
           kf f m n a1 a2 b #fA1 #fA2 #sa1 #sa2 #sb tid);
    frame = pure (SZ.fits (tlayout_ulen lb));
    teardown = teardown f m n nthr a1 a2 b #fA1 #fA2 #sa1 #sa2 #sb;
    setup    = setup    f m n nthr a1 a2 b #fA1 #fA2 #sa1 #sa2 #sb;
    kpre  = kpre #t f m n #la1 a1 #la2 a2 #lb b #fA1 #fA2 #sa1 #sa2 #sb;
    kpost = kpost #t f m n #la1 a1 #la2 a2 #lb b #fA1 #fA2 #sa1 #sa2 #sb;
    kpre_sendable = solve;
    kpost_sendable = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn row_broadcast2
  (#t : Type0) {| scalar t |}
  (f : t -> t -> t -> t)
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#la1 : layout1 m) {| ctlayout la1 |}
  (a1 : array1 t la1)
  (#la2 : layout1 m) {| ctlayout la2 |}
  (a2 : array1 t la2)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#_ : squash (is_global a1))
  (#_ : squash (is_global a2))
  (#_ : squash (is_global b))
  (#fA1 : perm)
  (#fA2 : perm)
  (#sa1 : chest1 t m)
  (#sa2 : chest1 t m)
  (#sb : chest2 t m n)
  norewrite
  preserves
    cpu ** on gpu_loc (a1 |-> Frac fA1 sa1) ** on gpu_loc (a2 |-> Frac fA2 sa2)
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_row_broadcast2 f sa1 sa2 sb)
{
  launch_sync (kdesc f m n a1 a2 b);
}
