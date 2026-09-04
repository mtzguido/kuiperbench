module Kuiper.KB.ConvT1DGeneral

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT
module C2 = Kuiper.KB.ConvT2DGeneral
module CS = Kuiper.KB.CheckedSize
module Map = Kuiper.Kernel.Map
module Chest = Kuiper.Chest

inline_for_extraction noextract
let const_zero_f32 (_ : f32) : f32 = zero

let map_const_zero (#n : nat) (s : chest1 f32 n)
  : Lemma (chest_map const_zero_f32 s == mk1 (fun _ -> zero))
  = Chest.lemma_equal_intro
      (chest_map const_zero_f32 s) (mk1 (fun _ -> zero));
    Chest.ext (chest_map const_zero_f32 s) (mk1 (fun _ -> zero))

inline_for_extraction noextract
fn guard_convt1d_raw_size
  (b cin l_in cout k s : szp)
  (p opad : sz)
  (d : szp)
  norewrite
  requires emp
  ensures pure (convt1d_raw_size_req b cin l_in cout k s p opad d)
{
  dguard ((opad <^ s) || (opad <^ d));
  let lm1 : sz = SZ.(l_in -^ 1sz);
  let km1 : sz = SZ.(k -^ 1sz);
  let ls = CS.mul s lm1;
  let dk = CS.mul d km1;
  let sum0 = CS.add ls dk;
  let sum1 = CS.add sum0 opad;
  let pos = CS.add sum1 1sz;
  let two_p = CS.mul 2sz p;
  dguard (two_p <^ pos);
  let l0 = C2.convt_out_dim l_in s d k p opad;
  let l_out : szp = l0;
  assert pure (SZ.v l_out == convt1d_out_len l_in s d k p opad);

  let _xlen = CS.mulp3 b cin l_in;
  let _klen = CS.mulp3 cin cout k;
  let ylen = CS.mulp3 b cout l_out;
  let _inner = CS.mulp cin k;
  let _crow = CS.mulp cout l_out;
  let _lp = CS.addp l_out p;
  let _kd = CS.mulp k d;
  let launch_bound : szp = max_blocks *^ max_threads;
  dguard (ylen <=^ launch_bound);
  assert pure (SZ.v ylen <= SZ.v launch_bound);
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
{
  guard_convt1d_raw_size b cin l_in cout k s p opad d;
  assert pure (convt1d_size_req b cin l_in cout k s p opad d);
  let l_out = C2.convt_out_dim l_in s d k p opad;
  assert pure (SZ.v l_out == convt1d_out_len l_in s d k p opad);
  let l_out : szp = l_out;

  let gbias = alloc0 #f32 cout (l1_forward cout);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 cout gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> mk1 (fun _ -> (zero #f32))));

  let gy = C2.convt2d_general_alloc_f32
    b cin 1sz l_in cout 1sz k 1sz s 0sz p 1sz d 1sz l_out
    gx gw gbias;
  with sy. assert (on gpu_loc (gy |-> sy));
  free gbias;
  { l_out = l_out; output = gy }
}

inline_for_extraction let () = ()
