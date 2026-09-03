module Kuiper.KB.RowScaleAlloc

(* Self-allocating [diag(a) @ b].  The private output receives a verified
   device-to-device copy of [b], then the packaged verified row-scale kernel
   updates that copy.  Thus the public inputs are preserved and the bridge
   neither clones nor stages tensor data. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout { to_seq, from_seq }
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
module RS = Kuiper.Kernel.RowScale
module KRS = Klas.RowScale
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn copy_row_major_f32
  (m n : szp)
  (src : array2 f32 (l2_row_major m n) { is_global src })
  (dst : array2 f32 (l2_row_major m n) { is_global dst })
  (#ss #sd : chest2 f32 m n)
  (#f : perm)
  preserves cpu ** on gpu_loc (src |-> Frac f ss)
  requires on gpu_loc (dst |-> sd)
  ensures on gpu_loc (dst |-> ss)
{
  let elems : szp = m *^ n;
  map_loc gpu_loc
    #(dst |-> sd)
    #(core dst |-> to_seq (l2_row_major m n) sd)
    fn _ { tensor_concr dst; };
  map_loc gpu_loc
    #(src |-> Frac f ss)
    #(core src |-> Frac f (to_seq (l2_row_major m n) ss))
    fn _ { tensor_concr src; };
  gpu_memcpy_device_to_device (core dst) (core src) elems;
  map_loc gpu_loc
    #(core src |-> Frac f (to_seq (l2_row_major m n) ss))
    #(src |-> Frac f ss)
    fn _ {
      tensor_abs (l2_row_major m n) (core src);
      rewrite (from_array (l2_row_major m n) (core src) |-> Frac f ss)
        as (src |-> Frac f ss);
    };
  map_loc gpu_loc
    #(core dst |-> to_seq (l2_row_major m n) ss)
    #(dst |-> ss)
    fn _ {
      tensor_abs (l2_row_major m n) (core dst);
      rewrite (from_array (l2_row_major m n) (core dst) |-> ss)
        as (dst |-> ss);
    }
}

fn row_scale_alloc_f32
  (m n : szp {
     SZ.fits (SZ.v m * SZ.v n) /\
     SZ.v m * SZ.v n <= max_blocks * max_threads })
  (a : array1 f32 (l1_forward m) { is_global a })
  (b : array2 f32 (l2_row_major m n) { is_global b })
  (#sa : chest1 f32 m)
  (#sb : chest2 f32 m n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (a |-> Frac fA sa) **
    on gpu_loc (b |-> Frac fB sb)
  returns out : array2 f32 (l2_row_major m n)
  ensures on gpu_loc (out |-> RS.s_row_scale sa sb)
{
  let out = alloc0 #f32 (m *^ n) (l2_row_major m n);
  copy_row_major_f32 m n b out;
  KRS.rowscale_f32_rowmajor m n a out;
  out
}
