module Kuiper.KB.Conv3DAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv3D
open Kuiper.Kernel.Conv3D.Naive
open Kuiper.KB.Conv3DGeneral
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

(* (a) Verified, extractable conv3d output-size formula, provably equal to the
   pure spec [(n + 2*pad - k) / stride + 1].  Conv3d here has dilation fixed to
   1 (the only mode the four upstream tests exercise), so the dilated kernel
   span equals [k].  The raw entry calls this once per spatial axis after its
   checked Pulse guards establish [k <= padded], preventing subtraction
   underflow, and [fits (n + 2*pad)], keeping the value in u32.  It is
   identical in shape to [conv2d_out_dim] from challenge50. *)
let conv3d_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)
  =
  let padded : sz = SZ.(n +^ (2sz *^ pad)) in
  SZ.(((padded -^ k) /^ stride) +^ 1sz)

(* Upper bound on the conv3d output dimension: [out <= n + 2*pad].  Since the
   kernel span [k >= 1] and stride [stride >= 1], we have
   [(padded - k)/stride <= padded - k <= padded - 1 < padded].  Mirror of
   [conv2d_out_dim_ub]. *)
let conv3d_out_dim_ub (n k stride pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ k <= n + 2 * pad)
          (ensures (n + 2 * pad - k) / stride + 1 <= n + 2 * pad)
  = let padded = n + 2 * pad in
    ML.lemma_div_mod (padded - k) stride;
    assert ((padded - k) / stride <= padded - k)

(* (b) Self-allocating entry point.  Allocates the [b*cout*d_out*h_out*w_out]
   output buffer on the GPU via [alloc0] (extracts to cudaMalloc), runs
   the verified [conv3d_general_f32], and RETURNS the freshly-allocated buffer
   directly (binding it to a let first would sever the separation-logic
   resource link).  The post forwards the full per-thread [conv3d_out_at]
   functional spec. *)
inline_for_extraction noextract
fn conv3d_general_alloc
  (b cin d_in h_in w_in cout kd kh kw : szp)
  (stride : szp)
  (pad : sz)
  (d_out h_out : szp)
  (w_out : szp { conv3d_size_req b cin d_in h_in w_in cout kd kh kw stride
                                  d_out h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kd * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * d_out * h_out * w_out))
  ensures
    (exists* (sy : chest1 f32 (b * cout * d_out * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * d_out * h_out * w_out}).
               acc1 sy tid ==
               conv3d_out_at b cin d_in h_in w_in cout kd kh kw stride pad
                             d_out h_out w_out sx sw sbias tid))
{
  (* All partial products of [b*cout*d_out*h_out*w_out] are bounded by the full
     product (every factor is [>= 1]), which fits per [conv3d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout) 1
    (SZ.v d_out * SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v d_out) 1
    (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v d_out * SZ.v h_out) 1
    w_out;
  let len_y : szp = SZ.(b *^ cout *^ d_out *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  conv3d_general_f32 b cin d_in h_in w_in cout kd kh kw stride pad
                     d_out h_out w_out gx gw gbias gy;
  gy
}

let conv3d_general_alloc_f32 =
  fun b cin d_in h_in w_in cout kd kh kw stride pad d_out h_out w_out
      gx gw gbias #fx #fw #fb #sx #sw #sbias ->
    conv3d_general_alloc b cin d_in h_in w_in cout kd kh kw stride pad
                         d_out h_out w_out gx gw gbias
                         #fx #fw #fb #sx #sw #sbias

inline_for_extraction noextract
fn guard_conv3d_raw_size
  (b cin d_in h_in w_in cout kd kh kw stride : szp)
  (pad : sz)
  norewrite
  requires emp
  ensures pure (conv3d_raw_size_req b cin d_in h_in w_in cout kd kh kw
    stride pad)
{
  let two_pad = CS.mul 2sz pad;
  let d_pad = CS.addp d_in two_pad;
  let h_pad = CS.addp h_in two_pad;
  let w_pad = CS.addp w_in two_pad;
  dguard (kd <=^ d_pad);
  dguard (kh <=^ h_pad);
  dguard (kw <=^ w_pad);
  let d0 = conv3d_out_dim d_in kd stride pad;
  let h0 = conv3d_out_dim h_in kh stride pad;
  let w0 = conv3d_out_dim w_in kw stride pad;
  let d_out : szp = d0;
  let h_out : szp = h0;
  let w_out : szp = w0;
  assert pure (SZ.v d_out == conv3d_out_len d_in kd stride pad);
  assert pure (SZ.v h_out == conv3d_out_len h_in kh stride pad);
  assert pure (SZ.v w_out == conv3d_out_len w_in kw stride pad);

  let _xlen = CS.mulp5 b cin d_in h_in w_in;
  let _klen = CS.mulp5 cout cin kd kh kw;
  let ylen = CS.mulp5 b cout d_out h_out w_out;
  let _inner = CS.mulp4 cin kd kh kw;
  let _kernel = CS.mulp3 kd kh kw;
  let _khw = CS.mulp kh kw;
  let _hw = CS.mulp h_out w_out;
  let _dhw = CS.mulp3 d_out h_out w_out;
  let _cdhw = CS.mulp4 cout d_out h_out w_out;
  let ds = CS.mulp d_out stride;
  let _dbound = CS.addp ds kd;
  let hs = CS.mulp h_out stride;
  let _hbound = CS.addp hs kh;
  let ws = CS.mulp w_out stride;
  let _wbound = CS.addp ws kw;
  let launch_bound : szp = max_blocks *^ max_threads;
  dguard (ylen <=^ launch_bound);
  assert pure (SZ.v ylen <= SZ.v launch_bound);
}

fn conv3d_raw_alloc_bias_f32
  (b cin d_in h_in w_in cout kd kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
    { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kd * kh * kw))
    { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kd * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) ** on gpu_loc (gbias |-> Frac fb sbias)
  returns r : conv3d_raw_result b cin d_in h_in w_in cout kd kh kw
    stride pad
  ensures conv3d_raw_post b cin d_in h_in w_in cout kd kh kw stride pad
    sx sw sbias r
{
  guard_conv3d_raw_size b cin d_in h_in w_in cout kd kh kw stride pad;
  let d0 = conv3d_out_dim d_in kd stride pad;
  let h0 = conv3d_out_dim h_in kh stride pad;
  let w0 = conv3d_out_dim w_in kw stride pad;
  assert pure (SZ.v d0 == conv3d_out_len d_in kd stride pad);
  assert pure (SZ.v h0 == conv3d_out_len h_in kh stride pad);
  assert pure (SZ.v w0 == conv3d_out_len w_in kw stride pad);
  let d_out : szp = d0;
  let h_out : szp = h0;
  let w_out : szp = w0;
  (* Call the concrete implementation directly.  Going through the
     eta-expanded extraction alias makes Pulse re-check its entire dependent
     function type at this application; for the five-dimensional shapes that
     creates thousands of otherwise trivial refinement goals. *)
  let gy = conv3d_general_alloc b cin d_in h_in w_in cout kd kh kw
    stride pad d_out h_out w_out gx gw gbias
    #fx #fw #fb #sx #sw #sbias;
  (| d_out, (| h_out, (| w_out, gy |) |) |)
}

fn conv3d_raw_alloc_zero_f32
  (b cin d_in h_in w_in cout kd kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * cin * d_in * h_in * w_in))
    { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kd * kh * kw))
    { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * d_in * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kd * kh * kw))
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns r : conv3d_raw_result b cin d_in h_in w_in cout kd kh kw
    stride pad
  ensures conv3d_raw_post b cin d_in h_in w_in cout kd kh kw stride pad
    sx sw (mk1 (fun _ -> (zero #f32))) r
{
  guard_conv3d_raw_size b cin d_in h_in w_in cout kd kh kw stride pad;
  let gbias = alloc0 #f32 cout (l1_forward cout);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 cout gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> mk1 (fun _ -> (zero #f32))));

  (* Spell out the dimension proof and call the allocation core directly.
     Routing through [conv3d_raw_alloc_bias_f32] would force Pulse to
     elaborate its nested dependent-pair postcondition only to project the
     same three dimensions again. *)
  let d0 = conv3d_out_dim d_in kd stride pad;
  let h0 = conv3d_out_dim h_in kh stride pad;
  let w0 = conv3d_out_dim w_in kw stride pad;
  assert pure (SZ.v d0 == conv3d_out_len d_in kd stride pad);
  assert pure (SZ.v h0 == conv3d_out_len h_in kh stride pad);
  assert pure (SZ.v w0 == conv3d_out_len w_in kw stride pad);
  let d_out : szp = d0;
  let h_out : szp = h0;
  let w_out : szp = w0;
  let gy = conv3d_general_alloc b cin d_in h_in w_in cout kd kh kw
    stride pad d_out h_out w_out gx gw gbias
    #fx #fw #_ #sx #sw #(mk1 (fun _ -> (zero #f32)));
  free gbias;
  (| d_out, (| h_out, (| w_out, gy |) |) |)
}
