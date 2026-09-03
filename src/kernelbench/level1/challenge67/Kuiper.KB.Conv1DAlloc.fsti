module Kuiper.KB.Conv1DAlloc

(* Self-allocating KernelBench L1 #67/#76 Conv1D surface.
   The ABI-facing [conv1d_raw_alloc_bias_f32] and
   [conv1d_raw_alloc_zero_f32] entries take raw dimensions, derive the output
   length (including dilation), create zero bias when needed, allocate the
   output, run the verified kernel, and return the length with the owned
   buffer.  The C++ bridge performs no convolution geometry, size validation,
   or GPU staging; invalid raw dimensions abort at checked Pulse guards.  The
   lower-level helpers remain internal building blocks. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv1D
open Kuiper.Kernel.Conv1D.Naive
open Kuiper.KB.Conv1DGeneral
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let conv1d_out_len
  (n : nat) (k stride dilation : pos) (pad : nat)
  : nat
  = let padded = n + 2 * pad in
    let eff_k = (k - 1) * dilation + 1 in
    if eff_k > padded then 0 else (padded - eff_k) / stride + 1

inline_for_extraction noextract
unfold
let conv1d_raw_size_req
  (b cin n cout : nat) (k stride : pos) (pad : nat) (dilation : pos)
  : prop
  = let l_out = conv1d_out_len n k stride dilation pad in
    SZ.fits (n + 2 * pad) /\
    (k - 1) * dilation + 1 <= n + 2 * pad /\
    conv1d_size_req b cin n cout k stride dilation l_out
(* (a) Verified, extractable conv1d output-size formula used by the raw entry
   (see .fst).  Conv1d
   here supports dilation, so the dilated kernel span is
   [eff_k = (k-1)*dilation + 1] and the output dimension is
   [(n + 2*pad - eff_k) / stride + 1].  The [requires]
   [(k-1)*dilation + 1 <= n + 2*pad] prevents size_t subtraction underflow. *)
val conv1d_out_dim (n k stride dilation : szp) (pad : sz)
  : Pure SZ.t
      (requires SZ.fits (SZ.v n + 2 * SZ.v pad) /\
                (SZ.v k - 1) * SZ.v dilation + 1 <= SZ.v n + 2 * SZ.v pad)
      (ensures fun r ->
         SZ.v r ==
         (SZ.v n + 2 * SZ.v pad - ((SZ.v k - 1) * SZ.v dilation + 1))
           / SZ.v stride + 1)

(* Upper bound on the conv1d output dimension: [out <= n + 2*pad] (see .fst). *)
val conv1d_out_dim_ub (n k stride dilation pad : nat)
  : Lemma (requires k >= 1 /\ stride >= 1 /\ dilation >= 1 /\
                    (k - 1) * dilation + 1 <= n + 2 * pad)
          (ensures
             (n + 2 * pad - ((k - 1) * dilation + 1)) / stride + 1
               <= n + 2 * pad)

(* (b) Self-allocating entry-point type.  Takes the raw conv1d dims plus
   [l_out] (supplied by the verified [conv1d_out_dim]).  Allocates the
   [b*cout*l_out] GPU output buffer, runs the verified kernel, and returns the
   buffer directly — ownership passes to the caller (the bridge wraps it in a
   torch tensor with a cudaFree deleter).  The post is the SAME per-thread
   [conv1d_out_at] functional spec the underlying kernel guarantees. *)
fn conv1d_general_alloc_f32
  (b cin l_in cout kk stride : szp)
(pad : sz)
(dilation : szp)
(l_out : szp { conv1d_size_req b cin l_in cout kk stride dilation l_out })
(gx : array1 f32 (l1_forward (b * cin * l_in))
     { is_global gx })
(gw : array1 f32 (l1_forward (cout * cin * kk))
     { is_global gw })
(gbias : array1 f32 (l1_forward cout)
     { is_global gbias })
(#fx #fw #fb : perm)
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

(* Complete public entries.  They derive [l_out], create the zero bias when
   needed, allocate the result, and return its verified length with the
   buffer.  Thus callers never perform convolution geometry or GPU staging. *)
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
