module Kuiper.KB.Conv1DGeneral

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv1D
open Kuiper.Kernel.Conv1D.Naive
inline_for_extraction noextract
fn conv1d_general_impl
  (#et : Type0) {| scalar et |}
  (b cin l_in cout kk : szp)
  (stride : szp)
  (pad : sz)
  (dilation : szp)
  (l_out : szp { conv1d_size_req b cin l_in cout kk stride dilation l_out })
  (gx : array1 et (l1_forward (b * cin * l_in))
        { is_global gx })
  (gw : array1 et (l1_forward (cout * cin * kk))
        { is_global gw })
  (gbias : array1 et (l1_forward cout)
        { is_global gbias })
  (gy : array1 et (l1_forward (b * cout * l_out))
        { is_global gy })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : erased (chest1 et (b * cin * l_in)))
  (#sw : erased (chest1 et (cout * cin * kk)))
  (#sbias : erased (chest1 et cout))
  (#sy0 : erased (chest1 et (b * cout * l_out)))
  norewrite
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    on gpu_loc (gy |-> sy0)
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 et (b * cout * l_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * l_out}).
               acc1 sy tid ==
               conv1d_out_at b cin l_in cout kk stride pad dilation
                             l_out sx sw sbias tid))
{
  conv1d_naive_gpu #et b cin l_in cout kk stride pad dilation l_out
                   gx gw gbias gy;
  ()
}

let conv1d_general_f32 : conv1d_general_ty f32 = conv1d_general_impl #f32

inline_for_extraction let () = ()
