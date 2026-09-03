module Kuiper.KB.ConvT3DGeneral

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Spec.ConvTranspose3D
open Kuiper.Spec.ConvTranspose2D { convT_out_len_1d }
open Kuiper.Kernel.ConvT3D.Naive
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas
module Map = Kuiper.Kernel.Map
module Chest = Kuiper.Chest
module CS = Kuiper.KB.CheckedSize

inline_for_extraction noextract
let const_zero_f32 (_ : f32) : f32 = zero

let map_const_zero (#n : nat) (s : chest1 f32 n)
  : Lemma (chest_map const_zero_f32 s == mk1 (fun _ -> zero))
  = Chest.lemma_equal_intro
      (chest_map const_zero_f32 s) (mk1 (fun _ -> zero));
    Chest.ext (chest_map const_zero_f32 s) (mk1 (fun _ -> zero))

(* (a) Verified, extractable ConvTranspose output-size formula (per spatial
   axis), provably equal to the PyTorch formula
   [(n-1)*s - 2*p + d*(k-1) + opad + 1].  The raw entry calls this after
   checked Pulse arithmetic establishes its preconditions.  It mirrors
   [convt_out_dim] from challenge57. *)
let convt_out_dim (n s d k : szp) (p opad : sz)
  : Pure SZ.t
      (requires
         SZ.fits ((SZ.v n - 1) * SZ.v s + SZ.v d * (SZ.v k - 1)
                  + SZ.v opad + 1) /\
         2 * SZ.v p <= (SZ.v n - 1) * SZ.v s + SZ.v d * (SZ.v k - 1)
                       + SZ.v opad + 1)
      (ensures fun r ->
         SZ.v r == (SZ.v n - 1) * SZ.v s - 2 * SZ.v p
                   + SZ.v d * (SZ.v k - 1) + SZ.v opad + 1)
  =
  let pos : sz = SZ.((n -^ 1sz) *^ s +^ d *^ (k -^ 1sz) +^ opad +^ 1sz) in
  SZ.(pos -^ (2sz *^ p))

(* Upper bound on the ConvTranspose output dimension (see challenge57). *)
let convt_out_dim_ub (n s d k p opad : nat)
  : Lemma (requires n >= 1 /\ k >= 1)
          (ensures (n - 1) * s - 2 * p + d * (k - 1) + opad + 1
                   <= (n - 1) * s + d * (k - 1) + opad + 1)
  = ()

inline_for_extraction noextract
fn convt3d_general_impl
  (#et : Type0) {| scalar et |}
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (sd sh sw : szp) (pd ph pw : sz) (dd dh dw : szp)
  (d_out h_out : szp)
  (w_out : szp { convT3d_size_req b cin d_in h_in w_in cout kd kh kw
                                  sd sh sw pd ph pw dd dh dw
                                  d_out h_out w_out })
  (gx : array1 et (l1_forward (b * cin * d_in * h_in * w_in))
        { is_global gx })
  (gw : array1 et (l1_forward (cin * cout * kd * kh * kw))
        { is_global gw })
  (gbias : array1 et (l1_forward cout)
        { is_global gbias })
  (gy : array1 et (l1_forward (b * cout * d_out * h_out * w_out))
        { is_global gy })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 et (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 et (cin * cout * kd * kh * kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b * cout * d_out * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 et (b * cout * d_out * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
               acc1 sy tid ==
               convT3d_out_at b cin d_in h_in w_in cout kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias tid))
{
  convt3d_naive_gpu #et b cin d_in h_in w_in cout kd kh kw sd sh sw
                    pd ph pw dd dh dw d_out h_out w_out gx gw gbias gy;
  ()
}

let convt3d_general_f32 = convt3d_general_impl #f32

(* (b) Self-allocating entry point.  Allocates the
   [b*cout*d_out*h_out*w_out] output buffer on the GPU via [alloc0]
   (extracts to cudaMalloc), runs the verified [convt3d_general_f32], and
   RETURNS the freshly-allocated buffer directly.  Mirrors
   [convt2d_general_alloc] from challenge57. *)
inline_for_extraction noextract
fn convt3d_general_alloc
  (b cin d_in h_in w_in cout : szp)
  (kd kh kw : szp)
  (sd sh sw : szp) (pd ph pw : sz) (dd dh dw : szp)
  (d_out h_out : szp)
  (w_out : szp { convT3d_size_req b cin d_in h_in w_in cout kd kh kw
                                  sd sh sw pd ph pw dd dh dw
                                  d_out h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kd * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out))
  ensures
    (exists* (sy : chest1 f32 (b * cout * d_out * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
               acc1 sy tid ==
               convT3d_out_at b cin d_in h_in w_in cout kd kh kw
                              sd sh sw pd ph pw dd dh dw
                              d_out h_out w_out sx sw_l sbias tid))
{
  (* All partial products of [b*cout*d_out*h_out*w_out] are bounded by the
     full product (every factor is [>= 1]), which fits per
     [convT3d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout)
                        1 (SZ.v d_out * SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v d_out)
                        1 (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v d_out * SZ.v h_out)
                        1 w_out;
  let len_y : szp = SZ.(b *^ cout *^ d_out *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  convt3d_general_f32 b cin d_in h_in w_in cout kd kh kw sd sh sw
                      pd ph pw dd dh dw d_out h_out w_out
                      gx gw gbias gy;
  gy
}

let convt3d_general_alloc_f32 =
  fun b cin d_in h_in w_in cout kd kh kw sd sh sw pd ph pw dd dh dw
      d_out h_out w_out gx gw gbias #fx #fw #fb #sx #sw_l #sbias ->
    convt3d_general_alloc b cin d_in h_in w_in cout kd kh kw sd sh sw
                          pd ph pw dd dh dw d_out h_out w_out
                          gx gw gbias
                          #fx #fw #fb #sx #sw_l #sbias

inline_for_extraction noextract
fn guard_convt3d_raw_size
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw opd oph opw : sz)
  (dd dh dw : szp)
  norewrite
  requires emp
  ensures pure (convt3d_raw_size_req b cin d_in h_in w_in cout kd kh kw
    sd sh sw pd ph pw opd oph opw dd dh dw)
{
  dguard ((opd <^ sd) || (opd <^ dd));
  dguard ((oph <^ sh) || (oph <^ dh));
  dguard ((opw <^ sw) || (opw <^ dw));

  let dm1 : sz = SZ.(d_in -^ 1sz);
  let kdm1 : sz = SZ.(kd -^ 1sz);
  let ds0 = CS.mul sd dm1;
  let ddk = CS.mul dd kdm1;
  let dsum0 = CS.add ds0 ddk;
  let dsum1 = CS.add dsum0 opd;
  let dpos = CS.add dsum1 1sz;
  let two_pd = CS.mul 2sz pd;
  dguard (two_pd <^ dpos);

  let hm1 : sz = SZ.(h_in -^ 1sz);
  let khm1 : sz = SZ.(kh -^ 1sz);
  let hs0 = CS.mul sh hm1;
  let hdk = CS.mul dh khm1;
  let hsum0 = CS.add hs0 hdk;
  let hsum1 = CS.add hsum0 oph;
  let hpos = CS.add hsum1 1sz;
  let two_ph = CS.mul 2sz ph;
  dguard (two_ph <^ hpos);

  let wm1 : sz = SZ.(w_in -^ 1sz);
  let kwm1 : sz = SZ.(kw -^ 1sz);
  let ws0 = CS.mul sw wm1;
  let wdk = CS.mul dw kwm1;
  let wsum0 = CS.add ws0 wdk;
  let wsum1 = CS.add wsum0 opw;
  let wpos = CS.add wsum1 1sz;
  let two_pw = CS.mul 2sz pw;
  dguard (two_pw <^ wpos);

  let d0 = convt_out_dim d_in sd dd kd pd opd;
  let h0 = convt_out_dim h_in sh dh kh ph oph;
  let w0 = convt_out_dim w_in sw dw kw pw opw;
  let d_out : szp = d0;
  let h_out : szp = h0;
  let w_out : szp = w0;
  assert pure (SZ.v d_out == convt3d_out_len d_in sd dd kd pd opd);
  assert pure (SZ.v h_out == convt3d_out_len h_in sh dh kh ph oph);
  assert pure (SZ.v w_out == convt3d_out_len w_in sw dw kw pw opw);

  let _xlen = CS.mulp5 b cin d_in h_in w_in;
  let _klen = CS.mulp5 cin cout kd kh kw;
  let ylen = CS.mulp5 b cout d_out h_out w_out;
  let _inner = CS.mulp4 cin kd kh kw;
  let _kernel = CS.mulp3 kd kh kw;
  let _khw = CS.mulp kh kw;
  let _hw = CS.mulp h_out w_out;
  let _dhw = CS.mulp3 d_out h_out w_out;
  let _cdhw = CS.mulp4 cout d_out h_out w_out;
  let _dp = CS.addp d_out pd;
  let _hp = CS.addp h_out ph;
  let _wp = CS.addp w_out pw;
  let _kdd = CS.mulp kd dd;
  let _khd = CS.mulp kh dh;
  let _kwd = CS.mulp kw dw;
  let launch_bound : szp = max_blocks *^ max_threads;
  dguard (ylen <=^ launch_bound);
  assert pure (SZ.v ylen <= SZ.v launch_bound);
}

fn convt3d_raw_alloc_bias_f32
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw opd oph opw : sz) (dd dh dw : szp)
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
    { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kd * kh * kw))
    { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) ** on gpu_loc (gbias |-> Frac fb sbias)
  returns r : convt3d_raw_result b cin d_in h_in w_in cout kd kh kw
    sd sh sw pd ph pw opd oph opw dd dh dw
  ensures convt3d_raw_post b cin d_in h_in w_in cout kd kh kw sd sh sw
    pd ph pw opd oph opw dd dh dw sx sw_l sbias r
{
  guard_convt3d_raw_size b cin d_in h_in w_in cout kd kh kw sd sh sw
    pd ph pw opd oph opw dd dh dw;
  let d0 = convt_out_dim d_in sd dd kd pd opd;
  let h0 = convt_out_dim h_in sh dh kh ph oph;
  let w0 = convt_out_dim w_in sw dw kw pw opw;
  assert pure (SZ.v d0 == convt3d_out_len d_in sd dd kd pd opd);
  assert pure (SZ.v h0 == convt3d_out_len h_in sh dh kh ph oph);
  assert pure (SZ.v w0 == convt3d_out_len w_in sw dw kw pw opw);
  let d_out : szp = d0;
  let h_out : szp = h0;
  let w_out : szp = w0;
  (* Avoid the eta-expanded extraction alias internally.  Applying that alias
     here makes Pulse reconstruct its large dependent type and produces
     thousands of duplicate shape obligations. *)
  let gy = convt3d_general_alloc b cin d_in h_in w_in cout kd kh kw
    sd sh sw pd ph pw dd dh dw d_out h_out w_out gx gw gbias
    #fx #fw #fb #sx #sw_l #sbias;
  { d_out = d_out; h_out = h_out; w_out = w_out; output = gy }
}

fn convt3d_raw_alloc_zero_f32
  (b cin d_in h_in w_in cout kd kh kw sd sh sw : szp)
  (pd ph pw opd oph opw : sz) (dd dh dw : szp)
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
    { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kd * kh * kw))
    { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kd * kh * kw))
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l)
  returns r : convt3d_raw_result b cin d_in h_in w_in cout kd kh kw
    sd sh sw pd ph pw opd oph opw dd dh dw
  ensures convt3d_raw_post b cin d_in h_in w_in cout kd kh kw sd sh sw
    pd ph pw opd oph opw dd dh dw sx sw_l
    (mk1 (fun _ -> (zero #f32))) r
{
  guard_convt3d_raw_size b cin d_in h_in w_in cout kd kh kw sd sh sw
    pd ph pw opd oph opw dd dh dw;
  let gbias = alloc0 #f32 cout (l1_forward cout);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 cout gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> mk1 (fun _ -> (zero #f32))));

  (* Avoid constructing and immediately destructing the bias entry's nested
     dependent result.  The three output-size equalities are small explicit
     proofs; the allocation core already has precisely the postcondition we
     need. *)
  let d0 = convt_out_dim d_in sd dd kd pd opd;
  let h0 = convt_out_dim h_in sh dh kh ph oph;
  let w0 = convt_out_dim w_in sw dw kw pw opw;
  assert pure (SZ.v d0 == convt3d_out_len d_in sd dd kd pd opd);
  assert pure (SZ.v h0 == convt3d_out_len h_in sh dh kh ph oph);
  assert pure (SZ.v w0 == convt3d_out_len w_in sw dw kw pw opw);
  let d_out : szp = d0;
  let h_out : szp = h0;
  let w_out : szp = w0;
  let gy = convt3d_general_alloc b cin d_in h_in w_in cout kd kh kw
    sd sh sw pd ph pw dd dh dw d_out h_out w_out gx gw gbias
    #fx #fw #_ #sx #sw_l #(mk1 (fun _ -> (zero #f32)));
  free gbias;
  { d_out = d_out; h_out = h_out; w_out = w_out; output = gy }
}

inline_for_extraction let () = ()
