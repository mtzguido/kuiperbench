module Kuiper.Kernel.Tril

(* Implementation of [Kuiper.Kernel.Tril].

   One thread per element.  Thread [tid] decodes [(row, col) = (tid / n,
   tid % n)], reads [b[row, col]], and writes [b[row, col]] unchanged on or
   below the diagonal, [zero] strictly above it.  Structure mirrors the
   verified [Kuiper.Kernel.RowScale] (in-place [Array2] elementwise kernel):
   a flat [kernel_desc_n] grid over [m * n] threads, with the [Array2]
   exploded into per-cell ownership and re-imploded in setup/teardown. *)

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
  (tid : natlt (m *^ n))
  : slprop
  = Cell b (tid_to_cell m n tid) |-> acc2 sb (tid / n) (tid % n)

unfold
let kpost
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (sb : chest2 t m n)
  (tid : natlt (m *^ n))
  : slprop
  = Cell b (tid_to_cell m n tid) |-> acc2 (s_tril sb) (tid / n) (tid % n)

ghost
fn setup
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#sb : chest2 t m n)
  ()
  norewrite
  requires
    b |-> sb
  ensures
    (forall+ (tid : natlt (m *^ n)).
      kpre m n b sb tid) **
    pure (SZ.fits (tlayout_ulen lb))
{
  tensor_ilower2 b;
  forevery_unfactor' (m *^ n) m n (fun r c ->
    Cell b (idx2 r c) |-> acc2 sb r c);
  forevery_ext #(natlt (m *^ n)) _ (kpre m n b sb);
  ()
}

ghost
fn teardown
  (#t : Type0) {| scalar t |}
  (m n : szp)
  (#lb : layout2 m n) {| ctlayout lb |}
  (b : array2 t lb)
  (#sb : chest2 t m n)
  ()
  norewrite
  requires
    (forall+ (tid : natlt (m *^ n)).
      kpost m n b sb tid) **
    pure (SZ.fits (tlayout_ulen lb))
  ensures
    b |-> s_tril sb
{
  forevery_ext #(natlt (m *^ n))
    (kpost m n b sb)
    (fun (tid : natlt (m *^ n)) ->
       Cell b (idx2 ((tid / n) <: natlt m) ((tid % n) <: natlt n))
         |-> acc2 (s_tril sb) (tid / n) (tid % n));
  forevery_factor' (m *^ n) m n (fun r c ->
    Cell b (idx2 r c) |-> acc2 (s_tril sb) r c);
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
  (tid : szlt (m *^ n))
  ()
  requires
    gpu **
    kpre m n b sb tid
  ensures
    gpu **
    kpost m n b sb tid
{
  let row : sz = tid /^ n; assert rewrites_to row (tid /^ n);
  let col : sz = tid %^ n; assert rewrites_to col (tid %^ n);
  let x = tensor_read_cell b ((row <: szlt m), ((col <: szlt n), ()));
  if (col <=^ row) {
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
                (ensures  b |-> s_tril sb)
  = {
    nthr = m *^ n;
    f = kf m n b #sb;
    frame = pure (SZ.fits (tlayout_ulen lb));
    teardown = teardown m n b #sb;
    setup    = setup    m n b #sb;
    kpre  = kpre #t m n #lb b sb;
    kpost = kpost #t m n #lb b sb;
    kpre_sendable = solve;
    kpost_sendable = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn tril
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
    on gpu_loc (b |-> s_tril sb)
{
  launch_sync (kdesc m n b);
}
