module Kuiper.KB.ConvT2DGeneral

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv2D
open Kuiper.Spec.ConvTranspose2D
open Kuiper.Kernel.ConvT2D.Naive
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
   checked Pulse arithmetic constructs the non-negative part
   [pos = (n-1)*s + d*(k-1) + opad + 1].  The only
   subtraction is [- 2*p], whose underflow is ruled out by the precondition
   [2*p <= pos].
   [SZ.fits pos] keeps the value in u32. *)
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

(* Upper bound on the ConvTranspose output dimension: the result never exceeds
   the non-negative part [pos], since the only subtraction [- 2*p] cannot
   increase it.  Mirror of [conv2d_out_dim_ub]. *)
let convt_out_dim_ub (n s d k p opad : nat)
  : Lemma (requires n >= 1 /\ k >= 1)
          (ensures (n - 1) * s - 2 * p + d * (k - 1) + opad + 1
                   <= (n - 1) * s + d * (k - 1) + opad + 1)
  = ()

inline_for_extraction noextract
fn convt2d_general_impl
  (#et : Type0) {| scalar et |}
  (b cin h_in w_in cout kh kw : szp)
  (sh sw : szp) (ph pw : sz) (dh dw : szp)
  (h_out : szp)
  (w_out : szp { convT2d_size_req b cin h_in w_in cout kh kw
                                  sh sw ph pw dh dw h_out w_out })
  (gx : array1 et (l1_forward (b * cin * h_in * w_in))
        { is_global gx })
  (gw : array1 et (l1_forward (cin * cout * kh * kw))
        { is_global gw })
  (gbias : array1 et (l1_forward cout)
        { is_global gbias })
  (gy : array1 et (l1_forward (b * cout * h_out * w_out))
        { is_global gy })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 et (b * cin * h_in * w_in))
  (#sw_l : chest1 et (cin * cout * kh * kw))
  (#sbias : chest1 et cout)
  (#sy0 : chest1 et (b * cout * h_out * w_out))
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  requires
    on gpu_loc (gy |-> sy0)
  ensures
    (exists* (sy : chest1 et (b * cout * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
               acc1 sy tid ==
               convT2d_out_at b cin h_in w_in cout kh kw
                              sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias tid))
{
  convt2d_naive_gpu #et b cin h_in w_in cout kh kw sh sw ph pw dh dw
                    h_out w_out gx gw gbias gy;
  ()
}

let convt2d_general_f32 = convt2d_general_impl #f32

(* (b) Self-allocating entry point.  Allocates the [b*cout*h_out*w_out] output
   buffer on the GPU via [alloc0] (extracts to cudaMalloc), runs the
   verified [convt2d_general_f32], and RETURNS the freshly-allocated buffer
   directly (binding it to a let first would sever the separation-logic
   resource link).  Mirrors [conv2d_general_alloc] from challenge50. *)
inline_for_extraction noextract
fn convt2d_general_alloc
  (b cin h_in w_in cout kh kw : szp)
  (sh sw : szp) (ph pw : sz) (dh dw : szp)
  (h_out : szp)
  (w_out : szp { convT2d_size_req b cin h_in w_in cout kh kw
                                  sh sw ph pw dh dw h_out w_out })
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in))
        { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kh * kw))
        { is_global gw })
  (gbias : array1 f32 (l1_forward cout)
        { is_global gbias })
  (#fx : perm) (#fw : perm) (#fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kh * kw))
  (#sbias : chest1 f32 cout)
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * h_out * w_out))
  ensures
    (exists* (sy : chest1 f32 (b * cout * h_out * w_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * h_out * w_out}).
               acc1 sy tid ==
               convT2d_out_at b cin h_in w_in cout kh kw
                              sh sw ph pw dh dw
                              h_out w_out sx sw_l sbias tid))
{
  (* All partial products of [b*cout*h_out*w_out] are bounded by the full
     product (every factor is [>= 1]), which fits per [convT2d_size_req]. *)
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout) 1 (SZ.v h_out * SZ.v w_out);
  ML.lemma_mult_le_left (SZ.v b * SZ.v cout * SZ.v h_out) 1 w_out;
  let len_y : szp = SZ.(b *^ cout *^ h_out *^ w_out);
  let gy = alloc0 #f32 len_y (l1_forward len_y);
  with em. assert (on gpu_loc (gy |-> em));
  convt2d_general_f32 b cin h_in w_in cout kh kw sh sw ph pw dh dw
                      h_out w_out gx gw gbias gy;
  gy
}

let convt2d_general_alloc_f32 =
  fun b cin h_in w_in cout kh kw sh sw ph pw dh dw h_out w_out
      gx gw gbias #fx #fw #fb #sx #sw_l #sbias ->
    convt2d_general_alloc b cin h_in w_in cout kh kw sh sw ph pw dh dw
                          h_out w_out gx gw gbias
                          #fx #fw #fb #sx #sw_l #sbias

inline_for_extraction noextract
fn guard_convt2d_raw_size
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw oph opw : sz)
  (dh dw : szp)
  norewrite
  requires emp
  ensures pure (convt2d_raw_size_req b cin h_in w_in cout kh kw sh sw
    ph pw oph opw dh dw)
{
  dguard ((oph <^ sh) || (oph <^ dh));
  dguard ((opw <^ sw) || (opw <^ dw));

  let hm1 : sz = SZ.(h_in -^ 1sz);
  let khm1 : sz = SZ.(kh -^ 1sz);
  let hs = CS.mul sh hm1;
  let hdk = CS.mul dh khm1;
  let hsum0 = CS.add hs hdk;
  let hsum1 = CS.add hsum0 oph;
  let hpos = CS.add hsum1 1sz;
  let two_ph = CS.mul 2sz ph;
  dguard (two_ph <^ hpos);

  let wm1 : sz = SZ.(w_in -^ 1sz);
  let kwm1 : sz = SZ.(kw -^ 1sz);
  let ws = CS.mul sw wm1;
  let wdk = CS.mul dw kwm1;
  let wsum0 = CS.add ws wdk;
  let wsum1 = CS.add wsum0 opw;
  let wpos = CS.add wsum1 1sz;
  let two_pw = CS.mul 2sz pw;
  dguard (two_pw <^ wpos);

  let h0 = convt_out_dim h_in sh dh kh ph oph;
  let w0 = convt_out_dim w_in sw dw kw pw opw;
  let h_out : szp = h0;
  let w_out : szp = w0;
  assert pure (SZ.v h_out == convt2d_out_len h_in sh dh kh ph oph);
  assert pure (SZ.v w_out == convt2d_out_len w_in sw dw kw pw opw);

  let _xlen = CS.mulp4 b cin h_in w_in;
  let _klen = CS.mulp4 cin cout kh kw;
  let ylen = CS.mulp4 b cout h_out w_out;
  let _inner = CS.mulp3 cin kh kw;
  let _kernel = CS.mulp kh kw;
  let _hw = CS.mulp h_out w_out;
  let _chw = CS.mulp3 cout h_out w_out;
  let _hp = CS.addp h_out ph;
  let _wp = CS.addp w_out pw;
  let _khd = CS.mulp kh dh;
  let _kwd = CS.mulp kw dw;
  let launch_bound : szp = max_blocks *^ max_threads;
  dguard (ylen <=^ launch_bound);
  assert pure (SZ.v ylen <= SZ.v launch_bound);
}

fn convt2d_raw_alloc_bias_f32
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw oph opw : sz) (dh dw : szp)
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kh * kw)) { is_global gw })
  (gbias : array1 f32 (l1_forward cout) { is_global gbias })
  (#fx #fw #fb : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kh * kw))
  (#sbias : chest1 f32 cout)
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l) ** on gpu_loc (gbias |-> Frac fb sbias)
  returns r :
    (ho : szp { SZ.v ho == convt2d_out_len h_in sh dh kh ph oph } &
     (wo : szp { SZ.v wo == convt2d_out_len w_in sw dw kw pw opw } &
      array1 f32 (l1_forward (b * cout * ho * wo))))
  ensures exists* (sy : chest1 f32
    (b * cout * (dfst r) * (dfst (dsnd r)))).
    on gpu_loc ((dsnd (dsnd r)) |-> sy) **
    pure (forall (tid : nat{tid < b * cout * (dfst r) * (dfst (dsnd r))}).
      acc1 sy tid == convT2d_out_at b cin h_in w_in cout kh kw sh sw ph pw
        dh dw (dfst r) (dfst (dsnd r)) sx sw_l sbias tid)
{
  guard_convt2d_raw_size b cin h_in w_in cout kh kw sh sw
    ph pw oph opw dh dw;
  let h0 = convt_out_dim h_in sh dh kh ph oph;
  let w0 = convt_out_dim w_in sw dw kw pw opw;
  assert pure (SZ.v h0 == convt2d_out_len h_in sh dh kh ph oph);
  assert pure (SZ.v w0 == convt2d_out_len w_in sw dw kw pw opw);
  let h_out : szp = h0;
  let w_out : szp = w0;
  let gy = convt2d_general_alloc_f32 b cin h_in w_in cout kh kw sh sw ph pw
    dh dw h_out w_out gx gw gbias;
  (| h_out, (| w_out, gy |) |)
}

fn convt2d_raw_alloc_zero_f32
  (b cin h_in w_in cout kh kw sh sw : szp)
  (ph pw oph opw : sz) (dh dw : szp)
  (gx : array1 f32 (l1_forward (b * cin * h_in * w_in)) { is_global gx })
  (gw : array1 f32 (l1_forward (cin * cout * kh * kw)) { is_global gw })
  (#fx #fw : perm)
  (#sx : chest1 f32 (b * cin * h_in * w_in))
  (#sw_l : chest1 f32 (cin * cout * kh * kw))
  norewrite
  preserves cpu ** on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw_l)
  returns r :
    (ho : szp { SZ.v ho == convt2d_out_len h_in sh dh kh ph oph } &
     (wo : szp { SZ.v wo == convt2d_out_len w_in sw dw kw pw opw } &
      array1 f32 (l1_forward (b * cout * ho * wo))))
  ensures exists* (sy : chest1 f32
    (b * cout * (dfst r) * (dfst (dsnd r)))).
    on gpu_loc ((dsnd (dsnd r)) |-> sy) **
    pure (forall (tid : nat{tid < b * cout * (dfst r) * (dfst (dsnd r))}).
      acc1 sy tid == convT2d_out_at b cin h_in w_in cout kh kw sh sw ph pw
        dh dw (dfst r) (dfst (dsnd r)) sx sw_l
        (mk1 (fun _ -> (zero #f32))) tid)
{
  guard_convt2d_raw_size b cin h_in w_in cout kh kw sh sw
    ph pw oph opw dh dw;
  let gbias = alloc0 #f32 cout (l1_forward cout);
  with ebias. assert (on gpu_loc (gbias |-> ebias));
  Map.map_gpu const_zero_f32 cout gbias;
  map_const_zero ebias;
  rewrite (on gpu_loc (gbias |-> chest_map const_zero_f32 ebias))
       as (on gpu_loc (gbias |-> mk1 (fun _ -> (zero #f32))));
  let r = convt2d_raw_alloc_bias_f32 b cin h_in w_in cout kh kw sh sw
    ph pw oph opw dh dw gx gw gbias;
  free gbias;
  r
}

inline_for_extraction let () = ()
