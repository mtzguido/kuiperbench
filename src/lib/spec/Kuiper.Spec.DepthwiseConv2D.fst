module Kuiper.Spec.DepthwiseConv2D

(* Implementation of the DepthwiseConv2D functional spec.  Mirrors
   [Kuiper.Spec.Conv2D.fst] minus the input-channel reduction axis. *)

open Kuiper
open Kuiper.Spec.Conv2D

let rec __dwconv2d_single
  (#et:Type) {| scalar et |}
  (#b_n #c_n #h_in #w_in : nat)
  (kh : pos) (kw : pos)
  (s_h : pos) (s_w : pos)
  (p_h : nat) (p_w : nat)
  (d_h : pos) (d_w : pos)
  (#h_out #w_out : nat)
  (x : etensor4 et b_n c_n h_in w_in)
  (weight : etensor4 et c_n 1 kh kw)
  (b : natlt b_n)
  (c : natlt c_n)
  (oh : natlt h_out)
  (ow : natlt w_out)
  (to : nat{to <= kh * kw})
  : GTot et (decreases to)
  = if to = 0 then zero
    else (
      let i = to - 1 in
      let kh_i : natlt kh = unrank_dw_kh kh kw i in
      let kw_i : natlt kw = unrank_dw_kw kh kw i in
      let h_idx : int = oh * s_h + kh_i * d_h - p_h in
      let w_idx : int = ow * s_w + kw_i * d_w - p_w in
      add
        (__dwconv2d_single kh kw s_h s_w p_h p_w d_h d_w
           x weight b c oh ow (to - 1))
        (mul (read_padded x b c h_idx w_idx)
             (tacc weight c 0 kh_i kw_i))
    )

let __dwconv2d_single_zero_lemma
  #et #b_n #c_n #h_in #w_in
  kh kw s_h s_w p_h p_w d_h d_w
  #h_out #w_out
  x weight b c oh ow
  = ()

let __dwconv2d_single_lemma
  #et #b_n #c_n #h_in #w_in
  kh kw s_h s_w p_h p_w d_h d_w
  #h_out #w_out
  x weight b c oh ow to
  = ()

let lemma_dwconv2d_index
  #et #b_n #c_n #h_in #w_in
  kh kw s_h s_w p_h p_w d_h d_w
  h_out w_out
  x weight bias b c oh ow
  = ()
