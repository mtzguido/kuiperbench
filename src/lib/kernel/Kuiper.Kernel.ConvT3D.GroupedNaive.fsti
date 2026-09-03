module Kuiper.Kernel.ConvT3D.GroupedNaive

(* Direct grouped ConvTranspose3D kernel. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Conv3D
open Kuiper.Spec.ConvTranspose3D
module SZ = Kuiper.SizeT

[@@erasable]
val lseq_to_t5
  (#et:Type) (d0 d1 d2 d3 d4 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3 * d4))
  : etensor5 et d0 d1 d2 d3 d4

val lseq_to_t5_index
  (#et:Type) (d0 d1 d2 d3 d4 : nat)
  (s : chest1 et (d0 * d1 * d2 * d3 * d4))
  (i:natlt d0) (j:natlt d1) (k:natlt d2) (l:natlt d3) (m:natlt d4)
  : Lemma (t5acc (lseq_to_t5 d0 d1 d2 d3 d4 s) i j k l m ==
           acc1 s ((((i * d1 + j) * d2 + k) * d3 + l) * d4 + m))
          [SMTPat (t5acc (lseq_to_t5 d0 d1 d2 d3 d4 s) i j k l m)]

let convT3d_grouped_out_at
  (#et:Type) {| scalar et |}
  (b : nat{b==8}) (cin_pg : pos{cin_pg==8}) (d_in : nat{d_in==12}) (h_in : nat{h_in==24}) (w_in : nat{w_in==48}) (cout_pg : pos{cout_pg==8})
  (kd kh kw : pos) (sd sh sw : pos) (pd ph pw : nat)
  (dd dh dw : pos) (d_out h_out w_out : nat)
  (sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (sw_lseq : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (sbias : chest1 et (4*cout_pg))
  (tid : nat{tid < b*(4*cout_pg)*d_out*h_out*w_out})
  : GTot et
  = let cout = 4 * cout_pg in
    let bi : natlt b = tid / (cout*d_out*h_out*w_out) in
    let r1 = tid % (cout*d_out*h_out*w_out) in
    let oc : natlt cout = r1 / (d_out*h_out*w_out) in
    let r2 = r1 % (d_out*h_out*w_out) in
    let od : natlt d_out = r2 / (h_out*w_out) in
    let r3 = r2 % (h_out*w_out) in
    let oh : natlt h_out = r3 / w_out in
    let ow : natlt w_out = r3 % w_out in
    convT3d_grouped_single 4 cin_pg cout_pg kd kh kw sd sh sw
      pd ph pw dd dh dw
      (lseq_to_t5 b (4*cin_pg) d_in h_in w_in sx)
      (lseq_to_t5 (4*cin_pg) cout_pg kd kh kw sw_lseq)
      (chest1_to_seq sbias) bi oc od oh ow

unfold
let convT3d_grouped_size_req
  (b cin_pg d_in h_in w_in cout_pg kd kh kw : nat)
  (sd sh sw : nat) (pd ph pw : nat) (dd dh dw : nat)
  (d_out h_out w_out : nat)
  : prop
  = SZ.fits (b * (4 * cin_pg) * d_in * h_in * w_in) /\
    SZ.fits ((4 * cin_pg) * cout_pg * kd * kh * kw) /\
    SZ.fits (b * (4 * cout_pg) * d_out * h_out * w_out) /\
    SZ.fits (cin_pg * kd * kh * kw) /\
    SZ.fits (kd * kh * kw) /\ SZ.fits (kh * kw) /\
    SZ.fits (h_out * w_out) /\ SZ.fits (d_out * h_out * w_out) /\
    SZ.fits (4 * cout_pg * d_out * h_out * w_out) /\
    SZ.fits (d_out + pd) /\ SZ.fits (h_out + ph) /\ SZ.fits (w_out + pw) /\
    SZ.fits (kd * dd) /\ SZ.fits (kh * dh) /\ SZ.fits (kw * dw) /\
    b * (4 * cout_pg) * d_out * h_out * w_out <= max_blocks * max_threads

inline_for_extraction noextract
fn convt3d_grouped_naive_gpu
  (#et : Type0) {| scalar et |}
  (b : szp{SZ.v b==8}) (cin_pg : szp{SZ.v cin_pg==8}) (d_in : szp{SZ.v d_in==12}) (h_in : szp{SZ.v h_in==24}) (w_in : szp{SZ.v w_in==48}) (cout_pg : szp{SZ.v cout_pg==8}) (kd : szp{SZ.v kd==3}) (kh : szp{SZ.v kh==5}) (kw : szp{SZ.v kw==7}) (sd : szp{SZ.v sd==2}) (sh : szp{SZ.v sh==2}) (sw : szp{SZ.v sw==2})
  (pd : sz{SZ.v pd==1}) (ph : sz{SZ.v ph==2}) (pw : sz{SZ.v pw==3}) (dd : szp{SZ.v dd==1}) (dh : szp{SZ.v dh==1}) (dw : szp{SZ.v dw==1}) (d_out : szp{SZ.v d_out==24}) (h_out : szp{SZ.v h_out==48}) (w_out : szp{SZ.v w_out==96})
  (#lx : layout1 (b * (4 * cin_pg) * d_in * h_in * w_in)) {| ctlayout lx |}
  (#lw : layout1 ((4 * cin_pg) * cout_pg * kd * kh * kw)) {| ctlayout lw |}
  (#lbias : layout1 (4 * cout_pg)) {| ctlayout lbias |}
  (#ly : layout1 (b * (4 * cout_pg) * d_out * h_out * w_out)) {| ctlayout ly |}
  (gx : array1 et lx) (gw : array1 et lw)
  (gbias : array1 et lbias) (gy : array1 et ly)
  (#sx : chest1 et (b*(4*cin_pg)*d_in*h_in*w_in))
  (#sw_l : chest1 et ((4*cin_pg)*cout_pg*kd*kh*kw))
  (#sbias : chest1 et (4*cout_pg))
  (#sy0 : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out))
  (#fx #fw #fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) ** on gpu_loc (gbias |-> Frac fb sbias)
  requires on gpu_loc (gy |-> sy0) **
    pure (is_global gx /\ is_global gw /\ is_global gbias /\ is_global gy /\
      convT3d_grouped_size_req b cin_pg d_in h_in w_in cout_pg kd kh kw
                               sd sh sw pd ph pw dd dh dw d_out h_out w_out)
  ensures
    exists* (sy : chest1 et (b*(4*cout_pg)*d_out*h_out*w_out)).
      on gpu_loc (gy |-> sy) **
      pure (forall (tid : nat{tid < b*(4*cout_pg)*d_out*h_out*w_out}).
        acc1 sy tid ==
        convT3d_grouped_out_at b cin_pg d_in h_in w_in cout_pg kd kh kw
                               sd sh sw pd ph pw dd dh dw d_out h_out w_out
                               sx sw_l sbias tid)

inline_for_extraction let () = ()



