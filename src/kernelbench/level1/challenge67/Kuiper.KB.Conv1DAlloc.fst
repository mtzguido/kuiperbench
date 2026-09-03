module Kuiper.KB.Conv1DAlloc

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv1D
open Kuiper.Kernel.Conv1D.Naive
open Kuiper.KB.Conv1DGeneral
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

(* (a) Verified, extractable conv1d output-size formula, provably equal to the
   pure spec [(n + 2*pad - eff_k) / stride + 1] with the dilated kernel span
   [eff_k = (k-1)*dilation + 1].  The raw entry calls this after checked Pulse
   arithmetic establishes [eff_k <= padded], preventing size_t subtraction
   underflow.  This mirrors [conv2d_out_dim] with the added dilation factor;
   [fits (n + 2*pad)] keeps it in u32.  The trailing [+1] fits because
   [eff_k >= 1] gives [(padded - eff_k)/stride <= padded - 1], so the result
   is [<= padded]. *)
let conv1d_out_dim (n k stride dilation : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                (SZ.v k - 1) * SZ.v dilation + 1 <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r ==
         (SZ.v n + 2 * SZ.v pad - ((SZ.v k - 1) * SZ.v dilation + 1))
           / SZ.v stride + 1)
  =
  let padded : sz = SZ.(n +^ (2sz *^ pad)) in
  let eff_k : sz = SZ.((k -^ 1sz) *^ dilation +^ 1sz) in
  SZ.(((padded -^ eff_k) /^ stride) +^ 1sz)

(* Upper bound on the conv1d output dimension: [out <= n + 2*pad].  Since the
   dilated kernel span [eff_k >= 1] and stride [stride >= 1], we have
   [(padded - eff_k)/stride <= padded - eff_k <= padded - 1 < padded].
   Mirror of [conv2d_out_dim_ub]. *)
let conv1d_out_dim_ub (n k stride dilation pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ dilation >= 1 /\
                    (k - 1) * dilation + 1 <= n + 2 * pad)
          (ensures
             (n + 2 * pad - ((k - 1) * dilation + 1)) / stride + 1
               <= n + 2 * pad)
  = let padded = n + 2 * pad in
    let eff_k = (k - 1) * dilation + 1 in
    ML.lemma_div_mod (padded - eff_k) stride;
    assert ((padded - eff_k) / stride <= padded - eff_k)

(* (b) Self-allocating entry point.  Allocates the [b*cout*l_out] output buffer
   on the GPU via [alloc0] (extracts to cudaMalloc), runs the verified
   [conv1d_general_f32], and RETURNS the freshly-allocated buffer directly
   (binding it to a let first would sever the separation-logic resource link).
   The post forwards the full per-thread [conv1d_out_at] functional spec. *)
inline_for_extraction noextract
fn conv1d_general_alloc
  (b cin l_in cout kk : szp)
  (stride : szp)
  (pad : sz)
  (dilation : szp)
  (l_out : szp { conv1d_size_req b cin l_in cout kk stride dilation l_out })
  (gx : array1 f32 (l1_forward (b * cin * l_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kk))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 f32 (b * cin * l_in))
  (#sw : chest1 f32 (cout * cin * kk))
  (#sbias : chest1 f32 cout)
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * l_out))
  ensures
    (exists* (sy : chest1 f32 (b * cout * l_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * l_out}).
               acc1 sy tid ==
               conv1d_out_at b cin l_in cout kk stride pad dilation
                             l_out sx sw sbias tid))
{
  (* The partial product [b*cout] is bounded by the full product
     [b*cout*l_out] (every factor is [>= 1]), which fits per
     [conv1d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout) 1 l_out;
  let len_y : szp = SZ.(b *^ cout *^ l_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  conv1d_general_f32 b cin l_in cout kk stride pad dilation l_out
                     gx gw gbias gy;
  gy
}

let conv1d_general_alloc_f32 =
  fun b cin l_in cout kk stride pad dilation l_out
      gx gw gbias #fx #fw #fb #sx #sw #sbias ->
    conv1d_general_alloc b cin l_in cout kk stride pad dilation l_out
                         gx gw gbias #fx #fw #fb #sx #sw #sbias

inline_for_extraction noextract
fn guard_conv1d_raw_size
  (b cin l_in cout kk stride : szp)
  (pad : sz)
  (dilation : szp)
  norewrite
  requires emp
  ensures pure (conv1d_raw_size_req b cin l_in cout kk stride pad dilation)
{
  let two_pad = CS.mul 2sz pad;
  let padded = CS.addp l_in two_pad;
  let km1 : sz = SZ.(kk -^ 1sz);
  let dilated = CS.mul dilation km1;
  let eff_k = CS.add dilated 1sz;
  dguard (eff_k <=^ padded);
  let l0 = conv1d_out_dim l_in kk stride dilation pad;
  let l_out : szp = l0;
  assert pure (SZ.v l_out == conv1d_out_len l_in kk stride dilation pad);

  let _xlen = CS.mulp3 b cin l_in;
  let _klen = CS.mulp3 cout cin kk;
  let ylen = CS.mulp3 b cout l_out;
  let _inner = CS.mulp cin kk;
  let _crow = CS.mulp cout l_out;
  let ls = CS.mulp l_out stride;
  let kd = CS.mulp kk dilation;
  let _index_bound = CS.add ls kd;
  let launch_bound : szp = max_blocks *^ max_threads;
  dguard (ylen <=^ launch_bound);
  assert pure (SZ.v ylen <= SZ.v launch_bound);
}

fn conv1d_raw_alloc_bias_f32
  (b cin l_in cout kk stride : szp) (pad : sz) (dilation : szp)
  (gx : array1 f32 (l1_forward (b * cin * l_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kk)) { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * l_in))
  (#sw : chest1 f32 (cout * cin * kk))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns r :
    (lo : szp { SZ.v lo == conv1d_out_len l_in kk stride dilation pad }
     & array1 f32 (l1_forward (b * cout * lo)))
  ensures
    exists* (sy : chest1 f32 (b * cout * (dfst r))).
      on gpu_loc ((dsnd r) |-> sy) **
      pure (forall (tid : nat{tid < b * cout * (dfst r)}).
        acc1 sy tid ==
          conv1d_out_at b cin l_in cout kk stride pad dilation (dfst r)
            sx sw sbias tid)
{
  guard_conv1d_raw_size b cin l_in cout kk stride pad dilation;
  let l_out0 = conv1d_out_dim l_in kk stride dilation pad;
  assert pure (SZ.v l_out0 == conv1d_out_len l_in kk stride dilation pad);
  let l_out : szp = l_out0;
  let gy = conv1d_general_alloc_f32 b cin l_in cout kk stride pad dilation
    l_out gx gw gbias;
  (| l_out, gy |)
}

fn conv1d_raw_alloc_zero_f32
  (b cin l_in cout kk stride : szp) (pad : sz) (dilation : szp)
  (gx : array1 f32 (l1_forward (b * cin * l_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cout * cin * kk)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * l_in))
  (#sw : chest1 f32 (cout * cin * kk))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw)
  returns r :
    (lo : szp { SZ.v lo == conv1d_out_len l_in kk stride dilation pad }
     & array1 f32 (l1_forward (b * cout * lo)))
  ensures
    exists* (sy : chest1 f32 (b * cout * (dfst r))).
      on gpu_loc ((dsnd r) |-> sy) **
      pure (forall (tid : nat{tid < b * cout * (dfst r)}).
        acc1 sy tid ==
          conv1d_out_at b cin l_in cout kk stride pad dilation (dfst r)
            sx sw (mk1 (fun _ -> (zero #f32))) tid)
{
  guard_conv1d_raw_size b cin l_in cout kk stride pad dilation;
  let gbias = alloc0 #f32 cout (l1_forward cout);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 cout gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> mk1 (fun _ -> (zero #f32))));

  let r = conv1d_raw_alloc_bias_f32 b cin l_in cout kk stride pad dilation
    gx gw gbias;
  free gbias;
  r
}
