module Kuiper.Kernel.ConvT2D.GroupedNaive

(* Direct grouped ConvTranspose2D kernel. One thread computes one element of
   the full NCHW output and reduces only its output channel's input group. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv2D
open Kuiper.Spec.ConvTranspose2D
module SZ = Kuiper.SizeT

[@@erasable]
val lseq_to_t4
  (#et:Type) (d0 d1 d2 d3 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3))
  : etensor4 et d0 d1 d2 d3

val lseq_to_t4_index
  (#et:Type) (d0 d1 d2 d3 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3))
  (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3)
  : Lemma (tacc (lseq_to_t4 d0 d1 d2 d3 s) i j k l ==
           acc1 s (((i * d1 + j) * d2 + k) * d3 + l))
          [SMTPat (tacc (lseq_to_t4 d0 d1 d2 d3 s) i j k l)]

let convT2d_grouped_out_at
  (#et:Type) {| scalar et |}
  (b : nat{b==16}) (cin_pg : pos{cin_pg==8}) (h_in : nat{h_in==128}) (w_in : nat{w_in==256}) (cout_pg : pos{cout_pg==16})
  (kh kw : pos) (sh sw : pos) (ph pw : nat) (dh dw : pos)
  (h_out w_out : nat)
  (sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (sw_lseq : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (tid : nat{tid < b*(4*cout_pg)*h_out*w_out})
  : GTot et
  = let cout = 4 * cout_pg in
    let bi : natlt b = tid / (cout*h_out*w_out) in
    let r1 = tid % (cout*h_out*w_out) in
    let oc : natlt cout = r1 / (h_out*w_out) in
    let r2 = r1 % (h_out*w_out) in
    let oh : natlt h_out = r2 / w_out in
    let ow : natlt w_out = r2 % w_out in
    convT2d_grouped_single 4 cin_pg cout_pg kh kw sh sw ph pw dh dw
      (lseq_to_t4 b (4*cin_pg) h_in w_in sx)
      (lseq_to_t4 (4*cin_pg) cout_pg kh kw sw_lseq)
      (chest1_to_seq sbias) bi oc oh ow

unfold
let convT2d_grouped_size_req
  (b cin_pg h_in w_in cout_pg kh kw : nat)
  (sh sw : nat) (ph pw : nat) (dh dw : nat)
  (h_out w_out : nat)
  : prop
  = SZ.fits (b * (4 * cin_pg) * h_in * w_in) /\
    SZ.fits ((4 * cin_pg) * cout_pg * kh * kw) /\
    SZ.fits (b * (4 * cout_pg) * h_out * w_out) /\
    SZ.fits (cin_pg * kh * kw) /\ SZ.fits (kh * kw) /\
    SZ.fits (h_out * w_out) /\ SZ.fits (4 * cout_pg * h_out * w_out) /\
    SZ.fits (h_out + ph) /\ SZ.fits (w_out + pw) /\
    SZ.fits (kh * dh) /\ SZ.fits (kw * dw) /\
    b * (4 * cout_pg) * h_out * w_out <= max_blocks * max_threads

inline_for_extraction noextract
fn convt2d_grouped_naive_gpu
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==16}) (cin_pg : szp{SZ.v cin_pg==8}) (h_in : szp{SZ.v h_in==128}) (w_in : szp{SZ.v w_in==256}) (cout_pg : szp{SZ.v cout_pg==16}) (kh : szp{SZ.v kh==3}) (kw : szp{SZ.v kw==5}) (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==3})
  (ph : sz{SZ.v ph==1}) (pw : sz{SZ.v pw==2}) (dh : szp{SZ.v dh==2}) (dw : szp{SZ.v dw==1}) (h_out : szp{SZ.v h_out==257}) (w_out : szp{SZ.v w_out==766})
  (#lx : layout1 (b * (4 * cin_pg) * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 (4 * cout_pg)) {| ctlayout lbias |}
  (#ly : layout1 (b * (4 * cout_pg) * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx) (gw : array1 et lw)
  (gbias : array1 et lbias) (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*h_out*w_out))
  (#fx #fw #fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) ** on gpu_loc (gbias |-> Frac fb sbias)
  requires on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\ is_global gbias /\ is_global gy /\
      convT2d_grouped_size_req b cin_pg h_in w_in cout_pg kh kw
                               sh sw ph pw dh dw h_out w_out)
  ensures
    exists* (sy : chest1 et (b*(4*cout_pg)*h_out*w_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b*(4*cout_pg)*h_out*w_out}).
        acc1 sy tid ==
        convT2d_grouped_out_at b cin_pg h_in w_in cout_pg kh kw
                               sh sw ph pw dh dw h_out w_out sx sw_l sbias tid)

inline_for_extraction let () = ()




