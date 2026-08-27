module Kuiper.KB.Conv2DDilatedAsym

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2DDilated
open Kuiper.Kernel.Conv2D.Dilated
open Kuiper.Spec.Pool1D { pool_out_len_1d }
module SZ = Kuiper.SizeT

(* Verified, extractable output-dimension formula for a dilated conv axis:
     out = (in + 2*pad - (dil*(k-1)+1)) / stride + 1   (floor, clamped at 0)
   shared spec [pool_out_len_1d] (identical floor formula).  Computing this
   inside the verification boundary keeps the unverified bridge free of the
   output-shape arithmetic. *)
let conv2dd_out_dim_sz
  (l k s d : szp) (p : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d (SZ.v l) (SZ.v k) (SZ.v s) (SZ.v p) (SZ.v d))
  =
  let kspan  : sz = SZ.((d *^ (k -^ 1sz)) +^ 1sz) in
  let padded : sz = SZ.(l +^ (2sz *^ p)) in
  if SZ.(padded <^ kspan) then 0sz
  else SZ.(((padded -^ kspan) /^ s) +^ 1sz)

inline_for_extraction noextract
fn conv2d_dilated_asym_impl
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout kh kw : szp)
  (sh sw : szp)
  (ph pw : sz)
  (dh dw : szp)
  (h_out : szp)
  (w_out : szp { conv2dd_size_req b cin h_in w_in cout kh kw sh sw ph pw dh dw
                                  h_out w_out })
  (gx : array1 et (l1_forward (b * cin * h_in * w_in))
        { is_global gx })
  (gw : array1 et (l1_forward (cout * cin * kh * kw))
        { is_global gw })
  (gbias : array1 et (l1_forward cout)
        { is_global gbias })
  (gy : array1 et (l1_forward (b * cout * h_out * w_out))
        { is_global gy })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 et (b * cin * h_in * w_in))
  (#sw_ : chest1 et (cout * cin * kh * kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b * cout * h_out * w_out))
  norewrite
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b * cout * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
               acc1 sy tid ==
               conv2dd_out_at b cin h_in w_in cout kh kw sh sw ph pw dh dw
                              h_out w_out sx sw_ sbias tid))
{
  conv2d_dilated_gpu #et b cin h_in w_in cout kh kw sh sw ph pw dh dw
                     h_out w_out gx gw gbias gy;
  ()
}

let conv2d_dilated_asym_f32 : conv2d_dilated_asym_ty f32 =
  conv2d_dilated_asym_impl #f32

inline_for_extraction let () = ()
