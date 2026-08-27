module Kuiper.KB.MaxPool2D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout { is_full }
open Kuiper.Tensor.Layout.Alg { l2_row_major }
open Kuiper.Monoid.Reduce.F32 { cmonoid_fmax_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce, windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
open Kuiper.Tensor.Layout.BCMPages { l2_bcm_pages, c_l2_bcm_pages }
open Kuiper.Array2.Recast { recast_gpu }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module ML = FStar.Math.Lemmas

(* Verified, extractable computation of the PyTorch 1-D pool output length,
   provably equal to the pure spec [pool_out_len_1d].  The C bridge calls
   this (per axis) instead of re-implementing the formula in unverified C.
   [s], [k], [d] are szp (>= 1) so the spec's [s = 0] branch is dead and
   the SZ subtraction [padded -^ kspan] is guarded by the [padded <^ kspan]
   test exactly as in the spec. *)
let pool_out_len_1d_sz
  (l k s p d : szp)
  : Pure SZ.t
      (requires SZ.fits (SZ.v d * (SZ.v k - 1) + 1) /\
                SZ.fits (SZ.v l + 2 * SZ.v p))
      (ensures fun r ->
         SZ.v r == pool_out_len_1d l k s p d)
  =
  let kspan  : sz = SZ.((d *^ (k -^ 1sz)) +^ 1sz) in
  let padded : sz = SZ.(l +^ (2sz *^ p)) in
  if SZ.(padded <^ kspan) then 0sz
  else SZ.(((padded -^ kspan) /^ s) +^ 1sz)

inline_for_extraction noextract
fn maxpool2d_axis_fw
  (#t : Type0) {| scalar t |}
  (m_inst : Kuiper.Monoid.Reduce.cmonoid t)
  (k s p d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l    : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
  (#lin  : layout2 bc l)     {| ctlayout lin  |}
  (#lout : layout2 bc l_out) {| ctlayout lout |}
  (input  : array2 t lin  { is_global input  })
  (output : array2 t lout { is_global output })
  (#fIn  : perm)
  (#sx   : EM.chest2 t bc l)
  (#sout : EM.chest2 t bc l_out)
  preserves
    cpu **
    on gpu_loc (input  |-> Frac fIn sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (SZ.fits (SZ.v l_out * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.v bc * SZ.v l_out <= max_blocks * max_threads)
  ensures
    on gpu_loc (output |->
      windowreduce_result m_inst sx
        k s p d l_out)
{
  windowreduce m_inst k s p d bc l l_out input output
}

inline_for_extraction noextract
let maxpool2d_axis_fw_f32 =
  fun k s p d bc l l_out #_ #_ #_ #_ input output #fIn #sx #sout ->
    maxpool2d_axis_fw #f32 cmonoid_fmax_f32 k s p d bc l l_out input output
      #fIn #sx #sout

let maxpool2d_axis_fw_rm_f32 =
  fun k s p d bc l l_out input output #fIn #sx #sout ->
    maxpool2d_axis_fw_f32 k s p d bc l l_out
      #(l2_row_major bc l)     #_
      #(l2_row_major bc l_out) #_
      input output
      #fIn #sx #sout

(* Upper bound on the pool output length: [L_out <= L + 2P].  [kspan >= 1] and
   [S >= 1] give [(padded - kspan)/S <= padded - kspan <= padded - 1 < padded].
   This lets the self-allocating axis entry below discharge ALL the
   [l_out]-dependent overflow / launch-bound side conditions from bounds stated
   purely on the raw axis dims [(bc, l)], so the bridge never computes
   [l_out]. *)
let pool_out_len_1d_ub (l k s p d : nat)
  : Lemma (requires k >= 1 /\ s >= 1 /\ d >= 1)
          (ensures pool_out_len_1d l k s p d <= l + 2 * p)
  = let kspan = d * (k - 1) + 1 in
    let padded = l + 2 * p in
    if padded < kspan || s = 0 then ()
    else begin
      ML.lemma_div_mod (padded - kspan) s;
      assert ((padded - kspan) / s <= padded - kspan)
    end

(* [x*y <= x'*y'] from componentwise bounds. *)
let mul_le2 (x y x' y' : nat)
  : Lemma (requires x <= x' /\ y <= y') (ensures x * y <= x' * y')
  = ML.lemma_mult_le_right y x x';   (* x*y <= x'*y *)
    ML.lemma_mult_le_left x' y y'     (* x'*y <= x'*y' *)

(* [a <= a*b] when [b >= 1]. *)
let ge1_absorb (a b : nat)
  : Lemma (requires b >= 1) (ensures a <= a * b)
  = ML.lemma_mult_le_left a 1 b

(* The (H,W) "plane" product is dominated by the master (bc,H,W) product. *)
let plane_le (bcv ph2 pw2 : nat)
  : Lemma (requires bcv >= 1) (ensures ph2 * pw2 <= bcv * ph2 * pw2)
  = ML.lemma_mult_le_right ph2 1 bcv;            (* ph2 <= bcv*ph2 *)
    ML.lemma_mult_le_right pw2 ph2 (bcv * ph2)   (* ph2*pw2 <= (bcv*ph2)*pw2 *)

(* The pool output length is positive whenever the dilated window fits in the
   padded axis.  (In the [else] branch [pool_out_len_1d = .. /s + 1 >= 1].) *)
let pool_out_len_1d_pos (l k s p d : nat)
  : Lemma (requires k >= 1 /\ s >= 1 /\ d >= 1 /\ d * (k - 1) + 1 <= l + 2 * p)
          (ensures pool_out_len_1d l k s p d >= 1)
  = ()

(* [a*x*y <= a*x'*y'] from the componentwise bounds (grouping ((a*x)*y)). *)
let prod3_le (a x y x' y' : nat)
  : Lemma (requires x <= x' /\ y <= y') (ensures a * x * y <= a * x' * y')
  = ML.lemma_mult_le_left a x x';            (* a*x <= a*x' *)
    ML.lemma_mult_le_left (a * x) y y';       (* (a*x)*y <= (a*x)*y' *)
    ML.lemma_mult_le_right y' (a * x) (a * x') (* (a*x)*y' <= (a*x')*y' *)

(* [a*x*y == a*y*x] (used to reconcile the H-major and W-major groupings of
   the master bound [bc*padded_h*padded_w]). *)
let prod3_comm (a x y : nat)
  : Lemma (a * x * y == a * y * x)
  = ML.paren_mul_right a x y;   (* (a*x)*y == a*(x*y) *)
    ML.swap_mul x y;            (* x*y == y*x *)
    ML.paren_mul_right a y x    (* a*(y*x) == (a*y)*x *)

(* The (full) [l2_row_major] layout is full: it is built by [pack], whose
   result type [full_tlayout] carries the [is_full] refinement. *)
let l2_row_major_full (m n : nat)
  : Lemma (ensures is_full (l2_row_major m n))
          [SMTPat (is_full (l2_row_major m n))]
  = ()

(* Self-allocating per-axis pass.  Given the flattened row count [bc], the inner
   axis length [l], and the pool params, this computes [l_out] via the verified
   [pool_out_len_1d_sz], allocates the [(bc, l_out)] GPU output buffer (extracts
   to cudaMalloc), runs the fmax windowreduce, and returns BOTH [l_out] and the
   freshly allocated buffer.  Ownership of the buffer transfers to the caller
   (the bridge wraps it in a torch tensor with a cudaFree deleter, permutes, and
   feeds it to the next axis pass).  The bridge thus computes no output length
   and allocates no pool buffer; it only supplies the flattened row count and
   inner length and checks raw-dimension contracts. *)
inline_for_extraction noextract
fn maxpool2d_axis_alloc
  (k s p d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l : szp { SZ.fits (SZ.v bc * SZ.v l) })
  (input : array2 f32 (l2_row_major bc l) { is_global input })
  (#fIn : perm)
  (#sx  : EM.chest2 f32 bc l)
  preserves
    cpu **
    on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
    pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
    pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
    pure (SZ.fits ((SZ.v l + 2 * SZ.v p) * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.fits (SZ.v bc * (SZ.v l + 2 * SZ.v p))) **
    pure (SZ.v bc * (SZ.v l + 2 * SZ.v p) <= max_blocks * max_threads)
  returns r : (lo:sz { SZ.v lo == pool_out_len_1d l k s p d }
               & array2 f32 (l2_row_major bc lo))
  ensures
    on gpu_loc ((dsnd r) |->
      windowreduce_result cmonoid_fmax_f32 sx
        k s p d (dfst r)) **
    pure (SZ.v (dfst r) ==
            pool_out_len_1d l k s p d) **
    pure (is_global (dsnd r)) **
    pure (is_full_array (core (dsnd r)))
{
  let l_out = pool_out_len_1d_sz l k s p d;
  pool_out_len_1d_ub l k s p d;
  ML.lemma_mult_le_left bc l_out (SZ.v l + 2 * SZ.v p);
  ML.lemma_mult_le_right s l_out (SZ.v l + 2 * SZ.v p);
  let output = alloc0 #f32 (bc *^ l_out) (l2_row_major bc l_out);
  maxpool2d_axis_fw_rm_f32 k s p d bc l l_out input output;
  (| (l_out <: (lo:sz { SZ.v lo == pool_out_len_1d l k s p d })), output |)
}

let maxpool2d_axis_alloc_f32 =
  fun k s p d bc l input #fIn #sx ->
    maxpool2d_axis_alloc k s p d bc l input #fIn #sx

(* ── Single verified, transpose-free 2-D max-pool entry ──────────────

   [maxpool2d_full_alloc] folds the WHOLE separable 2-D max pool into one
   verified F*/Pulse call, eliminating the unverified PyTorch
   [.permute().contiguous()] that used to sit between the two
   [windowreduce] passes in the C++ bridge.

   Input is a row-major (B*C*H, W) view ([bc = B*C], inner = W).

   Pass 1 reduces the inner W axis (via [maxpool2d_axis_alloc]), producing a
   row-major (B*C*H, W_out) intermediate [mid].

   Pass 2 must reduce H, which is no longer the inner axis.  Instead of
   physically transposing, we [recast_gpu] the SAME [mid] bytes through the
   bespoke [flat_bcm B*C W_out H] layout — a flat (B*C*W_out, H) matrix whose
   element (R, j) is the byte the 3-index (b,i,w) view places at
   b=R/W_out, i=j, w=R%W_out.  [windowreduce] then reduces the (now inner) H
   axis strided, writing directly into a freshly allocated
   [flat_bcm B*C W_out H_out] buffer which is *physically* the row-major
   (B,C,H_out,W_out) result — so no permute-back is needed either.

   The intermediate [mid]/[mid2] share one backing array, which is freed
   after pass 2.  The returned (W_out, H_out, buffer) triple's buffer carries
   the pass-2 [windowreduce_result] post over some (B*C*W_out, H) matrix
   [sx2]; tying [sx2] to pass 1's [windowreduce_result] (the full
   [maxpool2d_post]) is left to a follow-up (milestone 2). *)
#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn maxpool2d_full_alloc
  (kh kw sh sw ph pw dh dw : szp)
  (bc : szp)
  (h w : szp)
  (#_ : squash (SZ.fits (SZ.v bc * SZ.v h)))
  (input : array2 f32 (l2_row_major (bc * h) w) { is_global input })
  (#fIn : perm)
  (#sx  : EM.chest2 f32 (bc * h) w)
  preserves
    cpu **
    on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (SZ.fits (SZ.v dw * (SZ.v kw - 1) + 1)) **
    pure (SZ.fits (SZ.v dh * (SZ.v kh - 1) + 1)) **
    pure (SZ.fits (SZ.v w + 2 * SZ.v pw)) **
    pure (SZ.fits (SZ.v h + 2 * SZ.v ph)) **
    pure (SZ.v dw * (SZ.v kw - 1) + 1 <= SZ.v w + 2 * SZ.v pw) **
    pure (SZ.v dh * (SZ.v kh - 1) + 1 <= SZ.v h + 2 * SZ.v ph) **
    pure (SZ.fits ((SZ.v w + 2 * SZ.v pw) * SZ.v sw + SZ.v kw * SZ.v dw)) **
    pure (SZ.fits ((SZ.v h + 2 * SZ.v ph) * SZ.v sh + SZ.v kh * SZ.v dh)) **
    pure (SZ.fits (SZ.v bc * (SZ.v h + 2 * SZ.v ph) * (SZ.v w + 2 * SZ.v pw))) **
    pure (SZ.v bc * (SZ.v h + 2 * SZ.v ph) * (SZ.v w + 2 * SZ.v pw)
            <= max_blocks * max_threads)
  returns r :
    (wo : sz { SZ.v wo == pool_out_len_1d w kw sw pw dw
               /\ SZ.v wo > 0 }
     & (ho : sz { SZ.v ho == pool_out_len_1d h kh sh ph dh
                  /\ SZ.v ho > 0 }
        & array2 f32 (l2_bcm_pages bc wo ho)))
  ensures
    (exists* (sx2 : EM.chest2 f32 (SZ.v bc * SZ.v (dfst r)) h).
       on gpu_loc ((dsnd (dsnd r)) |->
         windowreduce_result cmonoid_fmax_f32 sx2
           kh sh ph dh (dfst (dsnd r))))
{
  (* All [bcv]/[ph2]/[pw2]/[wov]/[hov]-style values below are written
     inline as [SZ.v _] expressions: they appear ONLY as ghost lemma
     arguments and in erased type positions, so nothing reaches the
     extracted C (binding them with [let] would emit dead
     mathematical-integer code that references undefined Prims/SizeT
     helpers). *)

  (* ── Pass-1 bounds (on raw dims) from the master (bc,H,W) product ── *)
  mul_le2 bc h bc (SZ.v h + 2 * SZ.v ph);
  ge1_absorb (SZ.v bc * (SZ.v h + 2 * SZ.v ph)) (SZ.v w + 2 * SZ.v pw);
  prod3_le bc h w (SZ.v h + 2 * SZ.v ph) (SZ.v w + 2 * SZ.v pw);
  prod3_le bc h (SZ.v w + 2 * SZ.v pw) (SZ.v h + 2 * SZ.v ph) (SZ.v w + 2 * SZ.v pw);

  let r1 = maxpool2d_axis_alloc kw sw pw dw (bc *^ h) w input;
  pool_out_len_1d_pos w kw sw pw dw;  (* wo > 0 *)
  pool_out_len_1d_ub  w kw sw pw dw;  (* wo <= pw2 *)
  let wo : (x:sz { SZ.v x == pool_out_len_1d w kw sw pw dw
                   /\ SZ.v x > 0 }) = dfst r1;

  pool_out_len_1d_pos h kh sh ph dh;  (* ho > 0 *)
  pool_out_len_1d_ub  h kh sh ph dh;  (* ho <= ph2 *)
  let ho : (x:sz { SZ.v x == pool_out_len_1d h kh sh ph dh
                   /\ SZ.v x > 0 }) = pool_out_len_1d_sz h kh sh ph dh;

  (* ── Pass-2 bounds ── *)
  prod3_le bc wo ho (SZ.v w + 2 * SZ.v pw) (SZ.v h + 2 * SZ.v ph);
  prod3_comm bc (SZ.v w + 2 * SZ.v pw) (SZ.v h + 2 * SZ.v ph);
  ge1_absorb (SZ.v bc * SZ.v wo) ho;
  prod3_le bc wo h (SZ.v w + 2 * SZ.v pw) (SZ.v h + 2 * SZ.v ph);
  prod3_comm bc wo h;              (* recast size eq *)
  plane_le bc (SZ.v h + 2 * SZ.v ph) (SZ.v w + 2 * SZ.v pw);
  mul_le2 h  wo (SZ.v h + 2 * SZ.v ph) (SZ.v w + 2 * SZ.v pw);
  mul_le2 ho wo (SZ.v h + 2 * SZ.v ph) (SZ.v w + 2 * SZ.v pw);
  ML.lemma_mult_le_right sh ho (SZ.v h + 2 * SZ.v ph);

  (* ── Recast pass-1 output to the BCM-pages H view (l2_bcm_pages is a
        full_tlayout, so [is_full] holds by its refinement). ── *)
  let mid2 = recast_gpu (l2_bcm_pages (SZ.v bc) (SZ.v wo) (SZ.v h)) (dsnd r1);

  (* ── Allocate the final (BCM-pages) output and run pass 2 over H ── *)
  let out = alloc0 #f32 ((bc *^ wo) *^ ho) (l2_bcm_pages (SZ.v bc) (SZ.v wo) (SZ.v ho));
  maxpool2d_axis_fw #f32 cmonoid_fmax_f32 kh sh ph dh (bc *^ wo) h ho
    #(l2_bcm_pages (SZ.v bc) (SZ.v wo) (SZ.v h))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc)) wo h)
    #(l2_bcm_pages (SZ.v bc) (SZ.v wo) (SZ.v ho))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc)) wo ho)
    mid2 out;

  (* ── Free the (shared) intermediate and return ── *)
  free mid2;
  (| wo, (| ho, out |) |)
}
#pop-options

let maxpool2d_full_alloc_f32 =
  fun kh kw sh sw ph pw dh dw bc h w #sq input #fIn #sx ->
    maxpool2d_full_alloc kh kw sh sw ph pw dh dw bc h w #sq input #fIn #sx
