module Kuiper.KB.ConvT1DGeneral

(* Verified ConvTranspose1D composition used by KernelBench L1 #64, #74,
   and #79.  ConvTranspose1D is the H=Kh=1 specialization of the verified
   ConvTranspose2D kernel.  This entry takes the original flat NCL input and
   (Cin,Cout,K) weight buffers directly; the logical rank embedding, output
   length, zero bias, output allocation, kernel call, and raw size validation
   all remain below the verification boundary. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Kernel.ConvT2D.Naive { convT2d_out_at, convT2d_size_req }
open Kuiper.Spec.ConvTranspose2D { convT_out_len_1d }
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let convt1d_out_len
  (l_in : nat) (s d k : pos) (p opad : nat)
  : nat
  = convT_out_len_1d l_in s p d k opad

inline_for_extraction noextract
let convt1d_via2d_out_at
  (b cin l_in cout : nat) (k s : pos) (p : nat) (d : pos)
  (l_out : pos)
  (sx : chest1 f32 (b * cin * l_in))
  (sw : chest1 f32 (cin * cout * k))
  (tid : nat{tid < b * cout * l_out})
  : GTot f32
  = convT2d_out_at b cin 1 l_in cout 1 k 1 s 0 p 1 d 1 l_out
      sx sw (mk1 (fun _ -> (zero #f32))) tid

inline_for_extraction noextract
unfold
let convt1d_size_req
  (b cin l_in cout : nat) (k s : pos) (p opad : nat) (d : pos)
  : prop
  = let l_out = convt1d_out_len l_in s d k p opad in
    convT2d_size_req b cin 1 l_in cout 1 k 1 s 0 p 1 d 1 l_out

inline_for_extraction noextract
unfold
let convt1d_raw_size_req
  (b cin l_in cout : nat) (k s : pos) (p opad : nat) (d : pos)
  : prop
  = (opad < s \/ opad < d) /\
    SZ.fits ((l_in - 1) * s + d * (k - 1) + opad + 1) /\
    2 * p < (l_in - 1) * s + d * (k - 1) + opad + 1 /\
    convt1d_size_req b cin l_in cout k s p opad d

noeq type convt1d_alloc_result
  (b cin l_in cout k s : szp) (p opad : sz) (d : szp) = {
  l_out : lo:szp { SZ.v lo == convt1d_out_len l_in s d k p opad };
  output : array1 f32 (l1_forward (b * cout * 1 * l_out));
}

fn convt1d_general_alloc_f32
  (b cin l_in cout k s : szp)
  (p opad : sz)
  (d : szp)
  (gx : array1 f32 (l1_forward (b * cin * l_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * k)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * l_in))
  (#sw : chest1 f32 (cin * cout * k))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns r : convt1d_alloc_result b cin l_in cout k s p opad d
  ensures
    exists* (sy : chest1 f32 (b * cout * 1 * r.l_out)).
      on gpu_loc (r.output |-> sy) **
      pure (forall (tid : nat{tid < b * cout * 1 * r.l_out}).
        acc1 sy tid ==
          convt1d_via2d_out_at b cin l_in cout k s p d r.l_out
            sx sw tid)

inline_for_extraction let () = ()
