module Kuiper.KB.Conv2DAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Kernel.Conv2D.Naive
open Kuiper.KB.Conv2DGeneral
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

(* (a) Verified, extractable conv output-size formula, provably equal to the
   pure spec [(n + 2*pad - k) / stride + 1].  The raw entry calls this after
   its checked Pulse guards establish [k <= padded], preventing size_t
   subtraction underflow, and [fits (n + 2*pad)], keeping the value in u32.
   This mirrors [pool_out_len_1d_sz] from challenge44.
   The trailing [+1] fits because [k >= 1] gives
   [(padded - k)/stride <= padded - 1], so the result is [<= padded]. *)
let conv2d_out_dim (n k stride : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                SZ.v k <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r == (SZ.v n + 2 * SZ.v pad - SZ.v k) / SZ.v stride + 1)
  =
  let padded : sz = SZ.(n +^ (2sz *^ pad)) in
  SZ.(((padded -^ k) /^ stride) +^ 1sz)

(* Upper bound on the conv output dimension: [out <= n + 2*pad].  Since the
   kernel span [k >= 1] and stride [stride >= 1], we have
   [(padded - k)/stride <= padded - k <= padded - 1 < padded].  Mirror of
   [pool_out_len_1d_ub]. *)
let conv2d_out_dim_ub (n k stride pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ k <= n + 2 * pad)
          (ensures (n + 2 * pad - k) / stride + 1 <= n + 2 * pad)
  = let padded = n + 2 * pad in
    ML.lemma_div_mod (padded - k) stride;
    assert ((padded - k) / stride <= padded - k)

(* (b) Self-allocating entry point.  Allocates the [b*cout*h_out*w_out] output
   buffer on the GPU via [alloc0] (extracts to cudaMalloc), runs the
   verified [conv2d_general_f32], and RETURNS the freshly-allocated buffer
   directly (binding it to a let first would sever the separation-logic
   resource link).  The post forwards the full per-thread [conv2d_out_at]
   functional spec. *)
inline_for_extraction noextract
fn conv2d_general_alloc
  (b cin h_in w_in cout kh kw : szp)
  (stride : szp)
  (pad : sz)
  (h_out : szp)
  (w_out : szp { conv2d_size_req b cin h_in w_in cout kh kw stride h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kh * kw))
  (#sbias : chest1 f32 cout)
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
  ensures
    (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
               acc1 sy tid ==
               conv2d_out_at b cin h_in w_in cout kh kw stride pad
                             h_out w_out sx sw sbias tid))
{
  (* All partial products of [b*cout*h_out*w_out] are bounded by the full
     product (every factor is [>= 1]), which fits per [conv2d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout) 1 (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v h_out) 1 w_out;
  let len_y : szp = SZ.(b *^ cout *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  conv2d_general_f32 b cin h_in w_in cout kh kw stride pad h_out w_out
                     gx gw gbias gy;
  gy
}

let conv2d_general_alloc_f32 =
  fun b cin h_in w_in cout kh kw stride pad h_out w_out
      gx gw gbias #fx #fw #fb #sx #sw #sbias ->
    conv2d_general_alloc b cin h_in w_in cout kh kw stride pad h_out w_out
                         gx gw gbias #fx #fw #fb #sx #sw #sbias

inline_for_extraction noextract
fn guard_conv2d_raw_size
  (b cin h_in w_in cout kh kw stride : szp)
  (pad : sz)
  norewrite
  requires emp
  ensures pure (conv2d_raw_size_req b cin h_in w_in cout kh kw stride pad)
{
  let two_pad = CS.mul 2sz pad;
  let h_pad = CS.addp h_in two_pad;
  let w_pad = CS.addp w_in two_pad;
  dguard (kh <=^ h_pad);
  dguard (kw <=^ w_pad);
  let h0 = conv2d_out_dim h_in kh stride pad;
  let w0 = conv2d_out_dim w_in kw stride pad;
  let h_out : szp = h0;
  let w_out : szp = w0;
  assert pure (SZ.v h_out == conv2d_out_len h_in kh stride pad);
  assert pure (SZ.v w_out == conv2d_out_len w_in kw stride pad);

  let _xlen = CS.mulp4 b cin h_in w_in;
  let _klen = CS.mulp4 cout cin kh kw;
  let _inner = CS.mulp3 cin kh kw;
  let hw = CS.mulp h_out w_out;
  let _chw = CS.mulp3 cout h_out w_out;
  let ylen = CS.mulp4 b cout h_out w_out;
  let hs = CS.mulp h_out stride;
  let _hbound = CS.addp hs kh;
  assert pure (SZ.fits (SZ.v h_out * SZ.v stride + SZ.v kh));
  let ws = CS.mulp w_out stride;
  let _wbound = CS.addp ws kw;
  assert pure (SZ.fits (SZ.v w_out * SZ.v stride + SZ.v kw));
  let launch_bound : szp = max_blocks *^ max_threads;
  dguard (ylen <=^ launch_bound);
  assert pure (SZ.v ylen <= SZ.v launch_bound);
}

fn conv2d_raw_alloc_bias_f32
  (b cin h_in w_in cout kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kh * kw)) { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves
    cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns r : conv2d_raw_result b cin h_in w_in cout kh kw stride pad
  ensures
    exists* (sy : chest1 f32
      (b * cout * r.h_out * r.w_out)).
      on gpu_loc (r.output |-> sy) **
      pure (forall (tid : nat{
        tid < b * cout * r.h_out * r.w_out}).
        acc1 sy tid == conv2d_out_at b cin h_in w_in cout kh kw stride pad
          r.h_out r.w_out sx sw sbias tid)
{
  guard_conv2d_raw_size b cin h_in w_in cout kh kw stride pad;
  let h0 = conv2d_out_dim h_in kh stride pad;
  let w0 = conv2d_out_dim w_in kw stride pad;
  assert pure (SZ.v h0 == conv2d_out_len h_in kh stride pad);
  assert pure (SZ.v w0 == conv2d_out_len w_in kw stride pad);
  let h_out : szp = h0;
  let w_out : szp = w0;
  let gy = conv2d_general_alloc_f32 b cin h_in w_in cout kh kw stride pad
    h_out w_out gx gw gbias;
  { h_out = h_out; w_out = w_out; output = gy }
}

fn conv2d_raw_alloc_zero_f32
  (b cin h_in w_in cout kh kw stride : szp) (pad : sz)
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kh * kw)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw : chest1 f32 (cout * cin * kh * kw))
  norewrite
  preserves
    cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns r : conv2d_raw_result b cin h_in w_in cout kh kw stride pad
  ensures
    exists* (sy : chest1 f32
      (b * cout * r.h_out * r.w_out)).
      on gpu_loc (r.output |-> sy) **
      pure (forall (tid : nat{
        tid < b * cout * r.h_out * r.w_out}).
        acc1 sy tid == conv2d_out_at b cin h_in w_in cout kh kw stride pad
          r.h_out r.w_out sx sw
          (mk1 (fun _ -> (zero #f32))) tid)
{
  guard_conv2d_raw_size b cin h_in w_in cout kh kw stride pad;
  let gbias = alloc0 #f32 cout (l1_forward cout);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 cout gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> mk1 (fun _ -> (zero #f32))));
  let r = conv2d_raw_alloc_bias_f32 b cin h_in w_in cout kh kw stride pad
    gx gw gbias;
  free gbias;
  r
}
