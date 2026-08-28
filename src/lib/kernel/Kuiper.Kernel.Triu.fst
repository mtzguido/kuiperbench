module Kuiper.Kernel.Triu

(* In-place upper-triangular mask, symmetric to Kuiper.Kernel.Tril.
   One thread per element keeps entries on or above the diagonal and writes
   zero below it. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor
module SZ = Kuiper.SizeT

unfold
let tid_to_cell (m n : nat) (tid : natlt (m * n))
  : abs (m @| n @| INil) =
  idx2 (tid / n) (tid % n)

unfold
let kpre
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (sb : chest2 t m n)
  (tid : natlt (m * n))
  : slprop
  = Cell b (tid_to_cell m n tid) |-> acc2 sb (tid / n) (tid % n)

unfold
let kpost
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (sb : chest2 t m n)
  (tid : natlt (m * n))
  : slprop
  = Cell b (tid_to_cell m n tid) |-> acc2 (s_triu sb) (tid / n) (tid % n)

ghost
fn setup
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (nthr : szp { SZ.v nthr == m * n })
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#sb : chest2 t m n)
  ()
  norewrite
  requires
    b |-> sb
  ensures
    (forall+ (tid : natlt nthr).
      kpre m n b sb tid) **
    pure (SZ.fits (tlayout_ulen lb))
{
  tensor_ilower2 b;
  forevery_unfactor' (m * n) m n (fun r c ->
    Cell b (idx2 r c) |-> acc2 sb r c);
  forevery_ext #(natlt (m * n)) _ (kpre m n b sb);
  forevery_rw_size (m * n) nthr;
  ()
}

ghost
fn teardown
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (nthr : szp { SZ.v nthr == m * n })
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#sb : chest2 t m n)
  ()
  norewrite
  requires
    (forall+ (tid : natlt nthr).
      kpost m n b sb tid) **
    pure (SZ.fits (tlayout_ulen lb))
  ensures
    b |-> s_triu sb
{
  forevery_rw_size nthr (m * n);
  forevery_ext #(natlt (m * n))
    (kpost m n b sb)
    (fun (tid : natlt (m * n)) ->
       Cell b (idx2 ((tid / n) <: natlt m) ((tid % n) <: natlt n))
         |-> acc2 (s_triu sb) (tid / n) (tid % n));
  forevery_factor' (m * n) m n (fun r c ->
    Cell b (idx2 r c) |-> acc2 (s_triu sb) r c);
  tensor_iraise2 b;
  ()
}

inline_for_extraction noextract
fn kf
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#sb : chest2 t m n)
  (tid : szlt (m * n))
  ()
  preserves gpu
  requires
    kpre m n b sb tid
  ensures
    kpost m n b sb tid
{
  let row : sz = tid /^ n; assert rewrites_to row (tid /^ n);
  let col : sz = tid %^ n; assert rewrites_to col (tid %^ n);
  let x = tensor_read_cell b ((row <: szlt m), ((col <: szlt n), ()));
  if (row <=^ col) {
    tensor_write_cell b ((row <: szlt m), ((col <: szlt n), ())) x;
  } else {
    tensor_write_cell b ((row <: szlt m), ((col <: szlt n), ())) zero;
  };
}

inline_for_extraction noextract
let kdesc
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#_ : squash (is_global b))
  (#sb : chest2 t m n)
  : kernel_desc (requires b |-> sb)
                (ensures  b |-> s_triu sb)
  = [@@inline_let] let nthr : (x : szp { SZ.v x == m * n }) = m *^ n in {
    nthr = nthr;
    f = (fun (tid : szlt nthr) -> kf m n b #sb tid);
    frame = pure (SZ.fits (tlayout_ulen lb));
    teardown = teardown m n nthr b #sb;
    setup    = setup    m n nthr b #sb;
    kpre  = kpre #t m n #lb b sb;
    kpost = kpost #t m n #lb b sb;
    kpre_sendable = solve;
    kpost_sendable = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn triu
  (t : Type0) {| scalar t |}
  (m n : szp)
  (#_ : squash (m * n <= max_blocks * max_threads))
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#_ : squash (is_global b))
  (#sb : chest2 t m n)
  preserves cpu
  requires
    on gpu_loc (b |-> sb)
  ensures
    on gpu_loc (b |-> s_triu sb)
{
  launch_sync (kdesc m n b);
}
