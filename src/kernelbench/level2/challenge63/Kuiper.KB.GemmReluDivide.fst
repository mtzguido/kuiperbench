module Kuiper.KB.GemmReluDivide

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, c_l1_forward, c_l2_row_major }
module EM = Kuiper.EMatrix
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module KS = Kuiper.Seq.Common
module Map = Kuiper.Kernel.Map
module BA = Kuiper.Kernel.BiasAdd
module P = Kuiper.Kernel.GEMM.Naive2

(* Bridges between flat array2 ownership and its zero-cost rank-2 tensor view,
   required to feed the (tensor-based) GEMM API while keeping the rest of the
   proof array2-based. *)
ghost
fn bridge_fwd
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (#f : perm) (#s : EM.chest2 et rows cols)
  preserves on gpu_loc (a |-> Frac f s)
{
  rewrite (on gpu_loc (a |-> Frac f s))
       as (on gpu_loc (a |-> Frac f s));
}

ghost
fn bridge_bwd
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : array2 et l) (#f : perm) (#s : EM.chest2 et rows cols)
  preserves on gpu_loc (a |-> Frac f s)
{
  rewrite (on gpu_loc (a |-> Frac f s))
       as (on gpu_loc (a |-> Frac f s));
}

(* Named division-by-constant step, so the [map_gpu] lambda and the spec-side
   reasoning refer to the same closure (avoids anonymous-lambda mismatch). *)
inline_for_extraction noextract
let div_by (#t:Type0) {| floating t |} (divisor : t) (v : t) : t = div v divisor

(* Per-(i,j) discharge of [gemm_relu_div_post].  Given the EXACT bias-add
   per-element fact for the flat index [i*out+j], the two pointwise maps
   ([relu] then [div_by divisor]) compose to the desired float expression.
   [bias_add_at_ij] rewrites the flat bias-add entry into [chest2] form. *)
#push-options "--z3rlimit 100"
let grd_row_aux
  (batch out : nat)
  (divisor : f32)
  (mm : EM.chest2 f32 batch out)
  (sbias : chest1 f32 out)
  (sy_b : chest1 f32 (batch * out))
  (hyp : squash
    (forall (tid:nat). tid < batch * out ==>
       acc1 sy_b tid == BA.bias_add_at batch out mm sbias tid))
  (i : natlt batch) (j : natlt out)
  : Lemma
      (ensures
        acc1 (chest_map (div_by divisor) (chest_map relu sy_b)) (i * out + j) ==
          div (relu (add (acc2 mm i j) (acc1 sbias j))) divisor)
  = BA.bias_add_at_ij batch out mm sbias i j;
    assert (acc1 (chest_map relu sy_b) (i * out + j) == relu (acc1 sy_b (i * out + j)));
    assert (acc1 (chest_map (div_by divisor) (chest_map relu sy_b)) (i * out + j)
            == div_by divisor (acc1 (chest_map relu sy_b) (i * out + j)))
#pop-options

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn gemm_relu_divide_f32_impl
  (batch : szp)
  (input : szp)
  (out : szp {
     SZ.v batch * SZ.v out <= max_blocks * max_threads /\
     SZ.fits (SZ.v batch * SZ.v input) /\
     SZ.fits (SZ.v input * SZ.v out) /\
     SZ.fits (SZ.v batch * SZ.v out) })
  (divisor : f32)
  (x    : array2 f32 (l2_row_major batch input) { is_global x    })
  (wt   : array2 f32 (l2_row_major input out)   { is_global wt   })
  (bias : array1 f32 (l1_forward out)                  { is_global bias })
  (y    : array1 f32 (l1_forward (SZ.v batch * SZ.v out))     { is_global y    })
  (#sx   : EM.chest2 f32 batch input)
  (#swt  : EM.chest2 f32 input out)
  (#sbias: chest1 f32 out)
  (#sy   : chest1 f32 (SZ.v batch * SZ.v out))
  preserves
    cpu **
    on gpu_loc (x    |-> sx) **
    on gpu_loc (wt   |-> swt) **
    on gpu_loc (bias |-> sbias)
  requires
    on gpu_loc (y    |-> sy)
  ensures
    (exists* (sy' : chest1 f32 (SZ.v batch * SZ.v out)).
       on gpu_loc (y |-> sy') **
       pure (gemm_relu_div_post divisor sx swt sbias sy'))
{
  (* Expose Seq.length / SZ.fits facts for the concrete layouts. *)

  (* Scratch GEMM output: gC = x @ wt  (batch × out). *)
  let gC = alloc0 #f32 (batch *^ out) (l2_row_major batch out);
  with sc0. assert on gpu_loc (gC |-> sc0);
  bridge_fwd x;
  bridge_fwd wt;
  bridge_fwd gC;

  (* Launch 1: exact GEMM.  comb2 ignores the old gC value, so the result is
     exactly [matmul sx swt] (matmul_is_gemm SMTPat). *)
  P.mmcomb_gpu_exact (MS.comb2 #f32)
    #batch #out #input
    (x) (wt) (gC);
  with eC'. assert on gpu_loc (gC |-> eC');
  bridge_bwd x;
  bridge_bwd wt;
  bridge_bwd gC;
  assert pure (reveal eC' == MS.matmul (reveal sx) (reveal swt));

  let mm : EM.chest2 f32 batch out =
    hide (MS.matmul (reveal sx) (reveal swt));
  assert pure (eC' == mm);

  (* Launch 2: broadcast bias-add, writing the flat output y. *)
  BA.bias_add_gpu batch out gC bias y;
  with sy_b. assert on gpu_loc (y |-> sy_b);
  assert pure (forall (tid:nat). tid < SZ.v batch * SZ.v out ==>
                 acc1 sy_b tid == BA.bias_add_at batch out (reveal mm) sbias tid);

  (* Launch 3: pointwise ReLU. *)
  Map.map_gpu relu (batch *^ out) #_ #(c_l1_forward _) y;

  (* Launch 4: pointwise divide by the constant. *)
  Map.map_gpu (div_by divisor) (batch *^ out) #_ #(c_l1_forward _) y;
  with sy'. assert on gpu_loc (y |-> sy');

  (* Discharge per-(i,j) [gemm_relu_div_post]. *)
  Classical.forall_intro_2
    (grd_row_aux batch out divisor (reveal mm) sbias sy_b ());

  free gC;
  ()
}
#pop-options

let gemm_relu_divide_f32 = gemm_relu_divide_f32_impl
