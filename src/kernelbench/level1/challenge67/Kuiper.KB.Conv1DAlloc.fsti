module Kuiper.KB.Conv1DAlloc

(* Self-allocating bridge layer for KernelBench L1 #67/#76 — Conv1D forward
   (general).  Moves TWO pieces of arithmetic/allocation that used to live in
   UNVERIFIED C++ inside the verification boundary:

     1. [conv1d_out_dim]: the conv1d output-size division
        [(n + 2*pad - eff_k) / stride + 1] with dilated kernel span
        [eff_k = (k-1)*dilation + 1], a VERIFIED extractable SZ-level
        formula.  The C bridge calls this instead of re-implementing the
        formula.

     2. [conv1d_general_alloc_f32]: a self-allocating entry point that
        allocates the [b*cout*l_out] output buffer on the GPU via
        [alloc0] (extracts to cudaMalloc), runs the verified
        [Kuiper.KB.Conv1DGeneral.conv1d_general_f32], and returns the
        freshly-allocated buffer DIRECTLY.  Its post forwards the FULL
        functional [conv1d_out_at] spec the underlying kernel guarantees. *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Conv1D
open Kuiper.Kernel.Conv1D.Naive
open Kuiper.KB.Conv1DGeneral
module SZ = Kuiper.SizeT
(* (a) Verified, extractable conv1d output-size formula (see .fst).  Conv1d
   here supports dilation, so the dilated kernel span is
   [eff_k = (k-1)*dilation + 1] and the output dimension is
   [(n + 2*pad - eff_k) / stride + 1].  The [requires]
   [(k-1)*dilation + 1 <= n + 2*pad] is exactly the C-side "padded input >=
   effective kernel" check (so the size_t subtraction does not underflow). *)
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
inline_for_extraction noextract
type conv1d_general_alloc_ty =
  fn
  (b : szp)
  (cin : szp)
  (l_in : szp)
  (cout : szp)
  (kk : szp)
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
  (#sx : erased (chest1 f32 (b * cin * l_in)))
  (#sw : erased (chest1 f32 (cout * cin * kk)))
  (#sbias : erased (chest1 f32 cout))
  requires
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias)
  returns gy : array1 f32 (l1_forward (b * cout * l_out))
  ensures
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gw |-> Frac fw sw) **
    on gpu_loc (gbias |-> Frac fb sbias) **
    (exists* (sy : chest1 f32 (b * cout * l_out)).
       on gpu_loc (gy |-> sy) **
       pure (forall (tid : nat{tid < b * cout * l_out}).
               acc1 sy tid ==
               conv1d_out_at b cin l_in cout kk stride pad dilation
                             l_out sx sw sbias tid))

val conv1d_general_alloc_f32 : conv1d_general_alloc_ty
