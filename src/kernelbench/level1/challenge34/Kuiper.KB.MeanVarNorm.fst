module Kuiper.KB.MeanVarNorm

#lang-pulse
open Kuiper
open Kuiper.Scalars.Ops
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Approximates.Base
open Kuiper.Spec.Frobenius
open Kuiper.Spec.MeanVarNorm
module SZ = Kuiper.SizeT
module SqrtApprox = Kuiper.KB.Compat.SqrtApprox

(* Proof-local description of the concrete floating intermediates.  The
   public spec is [row_mean_var_normalized], which contains no numerical
   existential. *)
let row_mean_var_normalized
  (#bd:nat) (sx sx' : Seq.lseq f32 bd)
  (off d : nat{off + d <= bd}) (eps inv_d : f32) : prop =
  exists (sum sum2 mean m2 var var_eps inv neg_mean_inv : f32).
    (let row : Seq.lseq f32 d = Seq.slice sx off (off + d) in
     sum  %~ rsum (to_real_seq row) /\
     sum2 %~ frobenius_sumsq_r (to_real_seq row) /\
     mean         == mul sum  inv_d /\
     m2           == mul sum2 inv_d /\
     var          == sub m2 (mul mean mean) /\
     var_eps      == add var eps /\
     inv          == rsqrt var_eps /\
     neg_mean_inv == sub zero (mul mean inv) /\
     Seq.slice sx' off (off + d) ==
       affine_result #f32 inv neg_mean_inv #d row /\
     (d > 0 ==>
       Kuiper.Spec.MeanVarNorm.row_mean_var_normalized sx sx' off d eps))

let mean_var_float_post
  (b d:nat) (eps inv_d:f32)
  (sx sx':Seq.lseq f32 (b*d)) : prop =
  forall (r:nat). r < b ==>
    r*d+d <= b*d /\ row_mean_var_normalized sx sx' (r*d) d eps inv_d
module Map = Kuiper.Kernel.Map
module HRed = Kuiper.Kernel.HReduce
module KS = Kuiper.Seq.Common

let row_mean_var_real_from_witnesses
  (#bd:nat) (d:pos)
  (sx sx':Seq.lseq f32 bd) (off:nat{off+d <= bd})
  (eps inv_d sum sum2 mean m2 var var_eps inv neg_mean_inv:f32)
  : Lemma
      (requires
        sum %~ rsum (to_real_seq (Seq.slice sx off (off+d))) /\
        sum2 %~ frobenius_sumsq_r (to_real_seq (Seq.slice sx off (off+d))) /\
        inv_d %~ (1.0R /. FStar.Real.of_int d) /\
        mean == mul sum inv_d /\ m2 == mul sum2 inv_d /\
        var == sub m2 (mul mean mean) /\ var_eps == add var eps /\
        inv == rsqrt var_eps /\ neg_mean_inv == sub zero (mul mean inv) /\
        Seq.slice sx' off (off+d) ==
          affine_result #f32 inv neg_mean_inv #d (Seq.slice sx off (off+d)) /\
        row_mean_var_domain sx off d eps)
      (ensures Kuiper.Spec.MeanVarNorm.row_mean_var_normalized sx sx' off d eps)
  = let row = to_real_seq (Seq.slice sx off (off+d)) in
    let rmean = mvn_mean_r #d row in
    let rm2 = mvn_m2_r #d row in
    let rarg = mvn_arg_r #d (to_real eps) row in
    a_mul sum inv_d (rsum row) (1.0R /. FStar.Real.of_int d);
    assert (mean %~ rmean);
    a_mul sum2 inv_d (frobenius_sumsq_r row)
      (1.0R /. FStar.Real.of_int d);
    assert (m2 %~ rm2);
    a_mul mean mean rmean rmean;
    sub_approx m2 (mul mean mean) rm2 (rmean *. rmean);
    to_real_ok eps;
    a_add var eps (rm2 -. rmean *. rmean) (to_real eps);
    assert (var_eps %~ rarg);
    SqrtApprox.rsqrt_approx var_eps rarg;
    let rinv : real = FStar.Math.Sqrt.rsqrt rarg in
    assert (inv %~ rinv);
    a_mul mean inv rmean rinv;
    sub_approx (zero #f32) (mul mean inv) 0.0R (rmean *. rinv);
    assert (neg_mean_inv %~ (0.0R -. rmean *. rinv));
    let aux (j:nat{j<d}) : Lemma
      (Seq.index (Seq.slice sx' off (off+d)) j %~
       Seq.index (mvn_row_result_r #d (to_real eps) row) j)
      = let x = Seq.index (Seq.slice sx off (off+d)) j in
        let rx = Seq.index row j in
        to_real_ok x;
        a_mul x inv rx rinv;
        a_add (mul x inv) neg_mean_inv (rx *. rinv)
          (0.0R -. rmean *. rinv)
    in
    Classical.forall_intro aux

let seq_map_id_eq (#a:Type) (s : Seq.seq a)
  : Lemma (Seq.equal (KS.seq_map id s) s)
  = ()

(* Pointwise square approximation lemma. *)
let sq_step_approx
  (#t:Type0) {| scalar t, real_like t |}
  (x : t) (r : real)
  : Lemma (requires v_approximates x r)
          (ensures  v_approximates (square x) (sq_step_r r))
  = a_mul x x r r

let sq_step_approx_forall (#t:Type0) {| scalar t, real_like t |} ()
  : Lemma (square #t %~ sq_step_r)
  = Classical.forall_intro_2
      (fun (xv:t) ->
         Classical.move_requires (sq_step_approx #t xv))

let blit_slice_left
  (#a:Type) (s1 s2 : Seq.seq a) (off1 off2 cnt lo hi : nat)
  : Lemma (requires off1 + cnt <= Seq.length s1 /\
                    off2 + cnt <= Seq.length s2 /\
                    lo <= hi /\ hi <= off1)
          (ensures
            Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) lo hi ==
            Seq.slice s1 lo hi)
  = Seq.lemma_eq_intro
      (Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) lo hi)
      (Seq.slice s1 lo hi)

let blit_slice_right
  (#a:Type) (s1 s2 : Seq.seq a) (off1 off2 cnt lo hi : nat)
  : Lemma (requires off1 + cnt <= Seq.length s1 /\
                    off2 + cnt <= Seq.length s2 /\
                    off1 + cnt <= lo /\ lo <= hi /\ hi <= Seq.length s1)
          (ensures
            Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) lo hi ==
            Seq.slice s1 lo hi)
  = Seq.lemma_eq_intro
      (Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) lo hi)
      (Seq.slice s1 lo hi)

let blit_slice_inside
  (#a:Type) (s1 s2 : Seq.seq a) (off1 off2 cnt : nat)
  : Lemma (requires off1 + cnt <= Seq.length s1 /\
                    off2 + cnt <= Seq.length s2)
          (ensures
            Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) off1 (off1 + cnt) ==
            Seq.slice s2 off2 (off2 + cnt))
  = Seq.lemma_eq_intro
      (Seq.slice (KS.seq_blit s1 off1 s2 off2 cnt) off1 (off1 + cnt))
      (Seq.slice s2 off2 (off2 + cnt))

let mv_blit_step_lemma
  (b d : nat)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (vi : nat)
  (inv neg_mean_inv : f32)
  : Lemma
    (requires
      vi < b /\
      Seq.slice sx_pre (vi * d) (b * d) ==
        Seq.slice sx (vi * d) (b * d) /\
      sx_post ==
        KS.seq_blit sx_pre (vi * d)
          (affine_result #f32 inv neg_mean_inv #d
             (Seq.slice sx_pre (vi * d) (vi * d + d))) 0 d)
    (ensures
      Seq.slice sx_post ((vi + 1) * d) (b * d) ==
        Seq.slice sx ((vi + 1) * d) (b * d) /\
      Seq.slice sx_post (vi * d) (vi * d + d) ==
        affine_result #f32 inv neg_mean_inv #d
          (Seq.slice sx (vi * d) (vi * d + d)) /\
      (forall (r : nat). r * d + d <= vi * d ==>
        Seq.slice sx_post (r * d) (r * d + d) ==
        Seq.slice sx_pre  (r * d) (r * d + d)))
  = let row_pre  : Seq.lseq f32 d =
      Seq.slice sx_pre (vi * d) (vi * d + d) in
    let row_orig : Seq.lseq f32 d =
      Seq.slice sx     (vi * d) (vi * d + d) in
    FStar.Seq.Properties.slice_slice sx_pre (vi * d) (b * d) 0 d;
    FStar.Seq.Properties.slice_slice sx     (vi * d) (b * d) 0 d;
    Seq.lemma_eq_intro row_pre row_orig;
    let result_seq : Seq.seq f32 =
      affine_result #f32 inv neg_mean_inv #d row_pre in
    blit_slice_inside #f32 sx_pre result_seq (vi * d) 0 d;
    Seq.lemma_eq_intro
      (Seq.slice sx_post (vi * d) (vi * d + d))
      result_seq;
    blit_slice_right #f32 sx_pre result_seq
      (vi * d) 0 d ((vi + 1) * d) (b * d);
    let len_tail = b * d - vi * d in
    FStar.Seq.Properties.slice_slice sx_pre (vi * d) (b * d) d len_tail;
    FStar.Seq.Properties.slice_slice sx     (vi * d) (b * d) d len_tail;
    Seq.lemma_eq_intro
      (Seq.slice sx_pre ((vi + 1) * d) (b * d))
      (Seq.slice sx ((vi + 1) * d) (b * d));
    let bd : nat = b * d in
    FStar.Math.Lemmas.lemma_mult_le_right d vi b;
    let vid : nat = vi * d in
    let sx_post' : Seq.lseq f32 bd = sx_post in
    let sx_pre'  : Seq.lseq f32 bd = sx_pre in
    let aux (r : nat{r * d + d <= bd}) : Lemma
      (requires r * d + d <= vid)
      (ensures Seq.slice sx_post' (r * d) (r * d + d) ==
               Seq.slice sx_pre'  (r * d) (r * d + d))
      = blit_slice_left #f32 sx_pre' result_seq
          vid 0 d (r * d) (r * d + d)
    in
    let aux2 (r : nat) : Lemma
      (requires r * d + d <= vid)
      (ensures r * d + d <= bd /\
               Seq.slice sx_post' (r * d) (r * d + d) ==
               Seq.slice sx_pre'  (r * d) (r * d + d))
      = aux r
    in
    Classical.forall_intro (Classical.move_requires aux2)

let mv_loop_step_lemma
  (b d : nat)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (vi : nat)
  : Lemma
    (requires
      vi < b /\
      Seq.slice sx_pre (vi * d) (b * d) ==
        Seq.slice sx (vi * d) (b * d) /\
      (exists (inv neg_mean_inv : f32).
         sx_post == KS.seq_blit sx_pre (vi * d)
           (affine_result #f32 inv neg_mean_inv #d
             (Seq.slice sx_pre (vi * d) (vi * d + d))) 0 d))
    (ensures
      vi * d + d <= b * d /\
      Seq.slice sx_post ((vi + 1) * d) (b * d) ==
        Seq.slice sx ((vi + 1) * d) (b * d) /\
      (forall (r : nat). r * d + d <= vi * d ==>
        Seq.slice sx_post (r * d) (r * d + d) ==
        Seq.slice sx_pre  (r * d) (r * d + d)))
  = FStar.Math.Lemmas.lemma_mult_le_right d (vi + 1) b;
    FStar.Seq.Properties.slice_slice sx_pre (vi * d) (b * d) 0 d;
    FStar.Seq.Properties.slice_slice sx     (vi * d) (b * d) 0 d;
    Seq.lemma_eq_intro
      (Seq.slice sx_pre (vi * d) (vi * d + d))
      (Seq.slice sx     (vi * d) (vi * d + d));
    let p (inv neg_mean_inv : f32) : prop =
      sx_post == KS.seq_blit sx_pre (vi * d)
        (affine_result #f32 inv neg_mean_inv #d
          (Seq.slice sx_pre (vi * d) (vi * d + d))) 0 d
    in
    let goal : prop =
      vi * d + d <= b * d /\
      Seq.slice sx_post ((vi + 1) * d) (b * d) ==
        Seq.slice sx ((vi + 1) * d) (b * d) /\
      (forall (r : nat). r * d + d <= vi * d ==>
        Seq.slice sx_post (r * d) (r * d + d) ==
        Seq.slice sx_pre  (r * d) (r * d + d))
    in
    let inner (inv neg_mean_inv : f32) : Lemma
      (requires p inv neg_mean_inv)
      (ensures goal)
      = mv_blit_step_lemma b d sx sx_pre sx_post vi inv neg_mean_inv
    in
    Classical.forall_to_exists #f32
      #(fun (inv : f32) -> exists (neg_mean_inv : f32). p inv neg_mean_inv)
      #goal
      (fun (inv : f32) ->
         Classical.forall_to_exists #f32 #(p inv) #goal
           (fun (neg_mean_inv : f32) ->
              Classical.move_requires (inner inv) neg_mean_inv))

#push-options "--z3rlimit 30 --fuel 2 --ifuel 2"
let row_mean_var_normalized_lift
  (b d : nat) (eps inv_d : f32)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (vi : nat)
  (r : nat)
  : Lemma
    (requires
      r < vi /\ vi <= b /\
      row_mean_var_normalized sx sx_pre (r * d) d eps inv_d /\
      Seq.slice sx_post (r * d) (r * d + d) ==
        Seq.slice sx_pre  (r * d) (r * d + d))
    (ensures
      r * d + d <= b * d /\
      row_mean_var_normalized sx sx_post (r * d) d eps inv_d)
  = ()
#pop-options

(* Lift the FIRST argument of row_mean_var_normalized: if sx and sx_alt
   agree on the input row slice [off..off+d), then the normalisation
   predicate holds with sx as the first argument too.  The same
   existential witnesses work because only Seq.slice sx off (off+d)
   — not the whole sx — appears in the predicate body. *)
#push-options "--z3rlimit 30 --fuel 2 --ifuel 2"
let rmnv_lift_input
  (#bd : nat) (d : nat) (off : nat{off+d<=bd}) (eps inv_d : f32)
  (sx sx_alt sx_post : Seq.lseq f32 bd)
  : Lemma
    (requires
      row_mean_var_normalized sx_alt sx_post off d eps inv_d /\
      Seq.slice sx off (off+d) == Seq.slice sx_alt off (off+d))
    (ensures row_mean_var_normalized sx sx_post off d eps inv_d)
  = ()
#pop-options

(* Extract a per-row slice equality from a suffix-slice equality, then
   apply rmnv_lift_input.  Avoids calling FStar.Seq.Properties.slice_slice
   directly inside the Pulse loop body (where the accumulated context
   makes typing preconditions difficult).
   suffix_lo must satisfy: vi*d+d <= suffix_lo <= bd. *)
#push-options "--z3rlimit 30"
let rmnv_lift_input_via_suffix
  (#bd : nat) (d : nat) (vi : nat) (suffix_lo : nat{vi * d + d <= suffix_lo /\ suffix_lo <= bd})
  (eps inv_d : f32)
  (sx sx_alt sx_post : Seq.lseq f32 bd)
  : Lemma
    (requires
      row_mean_var_normalized sx_alt sx_post (vi * d) d eps inv_d /\
      Seq.slice sx (vi * d) suffix_lo ==
        Seq.slice sx_alt (vi * d) suffix_lo)
    (ensures row_mean_var_normalized sx sx_post (vi * d) d eps inv_d)
  = (* slice_slice tells us:
         Seq.slice (Seq.slice sx     (vi*d) suffix_lo) 0 d == Seq.slice sx     (vi*d) (vi*d+d)
         Seq.slice (Seq.slice sx_alt (vi*d) suffix_lo) 0 d == Seq.slice sx_alt (vi*d) (vi*d+d)
       Combined with suffix equality, E-graph gives:
         Seq.slice sx (vi*d) (vi*d+d) == Seq.slice sx_alt (vi*d) (vi*d+d). *)
    FStar.Seq.Properties.slice_slice sx (vi * d) suffix_lo 0 d;
    FStar.Seq.Properties.slice_slice sx_alt (vi * d) suffix_lo 0 d;
    rmnv_lift_input d (vi * d) eps inv_d sx sx_alt sx_post
#pop-options

(* The domain only depends on the input row slice, so the loop's preserved
   suffix transports it to the current mutable buffer. *)
let row_mean_var_domain_via_suffix
  (#bd:nat) (d:pos) (vi:nat)
  (suffix_hi:nat{vi*d+d <= suffix_hi /\ suffix_hi <= bd})
  (eps:f32) (sx sx_alt:Seq.lseq f32 bd)
  : Lemma
      (requires
        row_mean_var_domain sx (vi*d) d eps /\
        Seq.slice sx (vi*d) suffix_hi ==
          Seq.slice sx_alt (vi*d) suffix_hi)
      (ensures row_mean_var_domain sx_alt (vi*d) d eps)
  = FStar.Seq.Properties.slice_slice sx (vi*d) suffix_hi 0 d;
    FStar.Seq.Properties.slice_slice sx_alt (vi*d) suffix_hi 0 d

let mv_prefix_le_lemma (d vi r : nat)
  : Lemma
    (requires r < vi)
    (ensures r * d + d <= vi * d)
  = FStar.Math.Lemmas.lemma_mult_le_right d (r + 1) vi

(* Quantified form: every row r before vi satisfies r*d+d <= vi*d.
   Used to strengthen mv_loop_step_lemma's slice-equality condition
   into a universally quantified form over r < vi, so Z3 can close the
   existential-substitution in the row_mean_var_normalized invariant
   step without a direct call to row_mean_var_normalized_lift_forall. *)
let mv_prefix_le_forall_lemma (d vi : nat)
  : Lemma (ensures forall (r : nat). r < vi ==> r * d + d <= vi * d)
  = let aux (r : nat) : Lemma
      (requires r < vi)
      (ensures r * d + d <= vi * d)
    = mv_prefix_le_lemma d vi r
    in
    Classical.forall_intro (Classical.move_requires aux)

(* For any row r < vi, if the *next* row fits (vi*d+d <= b*d) then
   row r also fits (r*d+d <= b*d).  The nonlinear step r+1 <= vi =>
   (r+1)*d <= vi*d is discharged by mv_prefix_le_lemma; the rest is
   linear and handled by Z3 automatically. *)
let row_bound_forall_lemma (b d vi : nat)
  : Lemma
    (requires vi * d + d <= b * d)
    (ensures forall (r : nat). r < vi ==> r * d + d <= b * d)
  = let aux (r : nat) : Lemma
      (requires r < vi)
      (ensures r * d + d <= b * d)
    = mv_prefix_le_lemma d vi r
      (* gives r*d+d <= vi*d; then vi*d <= vi*d+d-d <= b*d linearly *)
    in
    Classical.forall_intro (Classical.move_requires aux)

(* Universal bound: for EVERY row r < b, the row fits within b*d.
   Proved once in a clean context using lemma_mult_le_right so that
   calling this from the (large-context) Pulse loop body avoids Z3
   having to redo NIA inline for the off-refinement subtype check. *)
let row_lt_b_bound_forall_lemma (d b : nat)
  : Lemma (ensures forall (r : nat). r < b ==> r * d + d <= b * d)
  = let aux (r : nat{r < b}) : Lemma (ensures r * d + d <= b * d) =
      if d = 0 then ()
      else FStar.Math.Lemmas.lemma_mult_le_right d (r + 1) b
    in
    Classical.forall_intro aux

(* Transfer row_mean_var_normalized from sx_pre to sx_post for all rows
   r < vi, given that the slices are equal.  Done as a pure F* lemma
   with Classical.forall_intro so that Z3 works in a CLEAN context
   (few accumulated facts).  The single-instance case (row_mean_var_normalized_lift)
   unfolds the transparent predicate and uses slice equality; this wrapper
   quantifies it.

   The preconditions use SEPARATE foralls (not combined /\-foralls) so that
   calling this from the Pulse loop body avoids the combined-forall Z3
   trigger problem.  The TYPING of preconditions 3 and 4 uses
   row_lt_b_bound_forall_lemma's global fact to prove off-refinements. *)

(* Right-multiplication congruence, proven once in a clean default-rlimit
   context.  Used to inject [r*d == vi*d] as a ground fact in proofs that
   run under [], where Z3 does not reliably rediscover
   the nonlinear congruence from [r == vi] within a low rlimit. *)
let mul_right_cong (a b c : nat)
  : Lemma (requires a == b) (ensures a * c == b * c) = ()

#push-options "--z3rlimit 40"
let transfer_rmnv_forall
    (bd d vi : nat)
    (sx sx_pre sx_post : Seq.lseq f32 bd)
    (eps inv_d : f32)
  : Lemma
    (requires
      vi * d <= bd /\
      (forall (r : nat). r < vi ==> r * d + d <= bd) /\
      (forall (r : nat). r < vi ==> row_mean_var_normalized sx sx_pre (r * d) d eps inv_d) /\
      (forall (r : nat). r < vi ==>
         Seq.slice sx_post (r * d) (r * d + d) ==
         Seq.slice sx_pre  (r * d) (r * d + d)))
    (ensures
      forall (r : nat). r < vi ==>
        0 <= r * d /\
        r * d + d <= bd /\
        row_mean_var_normalized sx sx_post (r * d) d eps inv_d)
  = let aux (r : nat{r < vi}) : Lemma
        (ensures 0 <= r * d /\
                 r * d + d <= bd /\
                 row_mean_var_normalized sx sx_post (r * d) d eps inv_d) =
      (* [0 <= r*d] (typed over int, so [r*d] needs no nat-refinement to
         state it) makes the [off:nat] refinement of the third conjunct
         available via F*'s sequential l_and rule, dodging the nonlinear
         [r*d >= 0] subtyping query.  Discharged below with a linear
         ulib lemma rather than a higher z3rlimit. *)
      FStar.Math.Lemmas.nat_times_nat_is_nat r d;
      (* Z3 in clean context: precond 3 gives rmnv(sx_pre), which unfolds to
         [∃ w. Seq.slice sx_pre (r*d)(r*d+d) == affine_result w (Seq.slice sx (r*d)(r*d+d))].
         Precond 4 gives Seq.slice sx_post (r*d)(r*d+d) == Seq.slice sx_pre (r*d)(r*d+d).
         E-graph transitivity: Seq.slice sx_post (r*d)(r*d+d) == affine_result w ... ✓
         Same witnesses → rmnv(sx_post). ✓ *)
      ()
    in
    Classical.forall_intro aux
#pop-options

(* Extend the per-row normalisation invariant from vi rows to vi+1.
   Done as a pure F* lemma using Classical.forall_intro so that the
   proof is a structural case split in F*'s type theory, not a Z3
   forall-extension query (which tends to time out or trigger Z3
   LP-solver bugs when row_mean_var_normalized's existentials are
   involved).

   The ensures carries an explicit length bound r*d+d<=bd as its
   FIRST CONJUNCT so that the row_mean_var_normalized off-argument
   (r*d) can be typed without NIA in the ensures typing context:
   F*'s sequential l_and rule puts the first conjunct in scope when
   typechecking the second.

   The TWO PRECONDITION FORALLS are kept SEPARATE (not combined into
   a single /\-body forall) so that the Pulse call site does not need
   to produce a combined forall assertion.  Combining them in Pulse
   changes Z3's e-match trigger structure and causes the proof to fail,
   whereas two separate foralls each matched by Z3 independently work
   reliably.

   #push-options for z3rlimit here because the else-branch needs Z3
   to derive r=vi by LA from (r>=vi) + (r<vi+1), then close by
   E-graph congruence (r*d = vi*d). *)
#push-options "--z3rlimit 40 --fuel 0 --ifuel 0"
let extend_row_norm_forall
    (bd d b vi : nat)
    (sx sx' : Seq.lseq f32 bd)
    (eps inv_d : f32)
  : Lemma
    (requires
      vi < b /\
      0 < d /\
      bd = b * d /\
      vi * d + d <= bd /\
      (forall (r : nat). r < vi ==> r * d + d <= bd) /\
      (forall (r : nat). r < vi ==> row_mean_var_normalized sx sx' (r * d) d eps inv_d) /\
      row_mean_var_normalized sx sx' (vi * d) d eps inv_d)
    (ensures
      forall (r : nat). r < vi + 1 ==>
        0 <= r * d /\
        r * d + d <= bd /\
        row_mean_var_normalized sx sx' (r * d) d eps inv_d)
  = let aux (r : nat{r < vi + 1}) : Lemma
        (ensures 0 <= r * d /\
                 r * d + d <= bd /\
                 row_mean_var_normalized sx sx' (r * d) d eps inv_d) =
      (* [0 <= r*d] first (typed over int) brings the [off:nat] refinement
         of the third conjunct into scope without the nonlinear subtyping
         query; discharged with a linear ulib lemma instead of a high
         z3rlimit. *)
      FStar.Math.Lemmas.nat_times_nat_is_nat r d;
      if r < vi then begin
        (* Case r < vi:
           Bound: precondition 5 instantiated at r gives r*d+d<=bd. \u2713
           RMV:   precondition 6 instantiated at r gives row_mean_var_normalized. \u2713
           (Z3 instantiates both separate foralls independently.) *)
        ()
      end else begin
        (* Case r = vi (else-branch r>=vi, refinement r<vi+1).  Establish
           r=vi explicitly, then inject the congruence r*d = vi*d as a ground
           fact so the heavy row_mean_var_normalized precondition (at offset
           vi*d) transfers by rewriting under. *)
        assert (r == vi);
        mul_right_cong r vi d;
        ()
      end
    in
    Classical.forall_intro aux
#pop-options

let row_mean_var_normalized_lift_forall
  (b d : nat) (eps inv_d : f32)
  (sx sx_pre sx_post : Seq.lseq f32 (b * d))
  (vi : nat)
  : Lemma
    (requires
      vi <= b /\
      (forall (r : nat). r < vi ==>
         r * d + d <= b * d /\
         row_mean_var_normalized sx sx_pre (r * d) d eps inv_d) /\
      (* Restrict quantifier to nat{r*d+d <= b*d} so F* can typecheck
         the Seq.slice bounds without depending on the parallel vi <= b
         conjunct (l_and is not dependent at the type level). *)
      (forall (r : nat{r * d + d <= b * d}). r * d + d <= vi * d ==>
         Seq.slice sx_post (r * d) (r * d + d) ==
         Seq.slice sx_pre  (r * d) (r * d + d)))
    (ensures
      forall (r : nat). r < vi ==>
        r * d + d <= b * d /\
        row_mean_var_normalized sx sx_post (r * d) d eps inv_d)
  = let aux (r : nat) : Lemma
      (requires r < vi)
      (ensures r * d + d <= b * d /\
               row_mean_var_normalized sx sx_post (r * d) d eps inv_d)
    = mv_prefix_le_lemma d vi r;
      (* Provide vi*d <= b*d so the SMT can close r*d+d <= b*d
         from r*d+d <= vi*d  (established by mv_prefix_le_lemma). *)
      FStar.Math.Lemmas.lemma_mult_le_right d vi b;
      row_mean_var_normalized_lift b d eps inv_d sx sx_pre sx_post vi r
    in
    Classical.forall_intro (Classical.move_requires aux)

#push-options "--z3rlimit 30 --fuel 2 --ifuel 2"
let row_mean_var_normalized_intro
  (#bd : nat) (d : pos)
  (sx sx' : Seq.lseq f32 bd)
  (off : nat { off + d <= bd })
  (eps inv_d : f32)
  (sum sum2 mean m2 var var_eps inv neg_mean_inv : f32)
  : Lemma
      (requires
        sum  %~ rsum (to_real_seq (Seq.slice sx off (off + d))) /\
        sum2 %~ frobenius_sumsq_r (to_real_seq (Seq.slice sx off (off + d))) /\
        mean == mul sum inv_d /\
        m2 == mul sum2 inv_d /\
        var == sub m2 (mul mean mean) /\
        var_eps == add var eps /\
        inv == rsqrt var_eps /\
        neg_mean_inv == sub zero (mul mean inv) /\
        Seq.slice sx' off (off + d) ==
          affine_result #f32 inv neg_mean_inv #d
            (Seq.slice sx off (off + d)) /\
        inv_d %~ (1.0R /. FStar.Real.of_int d) /\
        row_mean_var_domain sx off d eps)
      (ensures row_mean_var_normalized sx sx' off d eps inv_d)
  = row_mean_var_real_from_witnesses d sx sx' off eps inv_d
      sum sum2 mean m2 var var_eps inv neg_mean_inv
#pop-options

(* Single combined lemma that re-establishes the host loop invariant after
   one iteration.  Mirrors L2Norm's [l2_loop_invariant_step]: performing all
   the forall-composition inside ONE pure F* lemma (a clean, narrow SMT
   context) avoids the fragile chest-congruence / existential-substitution
   problems that arise when these steps are attempted inline in the heavy
   Pulse loop body.  The requires are stated so they map DIRECTLY onto the
   loop invariant (conjuncts 3,4) and [mean_var_norm_row]'s ensures
   (conjuncts 5,6) at the call site, so the Pulse body needs no intermediate
   assertions -- only the [chest1_to_seq (reveal sx_pre) == chest1_to_seq sx']
   congruence, which fires reliably in the minimal post-[with] context. *)
#push-options "--z3rlimit 100"
let mv_loop_invariant_step
  (bd b d : nat)
  (sx sx_pre sx_post : Seq.lseq f32 bd)
  (vi : nat)
  (eps inv_d : f32)
  : Lemma
    (requires
      bd == b * d /\
      vi < b /\
      0 < d /\
      (forall (r : nat). r < vi ==>
         row_mean_var_normalized sx sx_pre (r * d) d eps inv_d) /\
      Seq.slice sx_pre (vi * d) bd ==
        Seq.slice sx (vi * d) bd /\
      row_mean_var_normalized sx_pre sx_post (vi * d) d eps inv_d /\
      (exists (inv neg_mean_inv : f32).
         sx_post == KS.seq_blit sx_pre (vi * d)
           (affine_result #f32 inv neg_mean_inv #d
             (Seq.slice sx_pre (vi * d) (vi * d + d))) 0 d))
    (ensures
      vi + 1 <= b /\
      (forall (r : nat). r < vi + 1 ==>
         row_mean_var_normalized sx sx_post (r * d) d eps inv_d) /\
      Seq.slice sx_post ((vi + 1) * d) bd ==
        Seq.slice sx ((vi + 1) * d) bd)
  =
    (* The seq parameters use an explicit flat length [bd], with
       [bd == b * d] as a pure hypothesis, so every predicate in the loop
       invariant has an identical type index.  Every [bd]-indexed helper
       below is passed [bd] directly; only
       [mv_loop_step_lemma] is stated over [b * d], and its seq arguments
       coerce here via that hypothesis inside this small lemma context. *)
    (* suffix preservation + prefix preservation + vi*d+d <= b*d *)
    mv_loop_step_lemma b d sx sx_pre sx_post vi;
    FStar.Math.Lemmas.lemma_mult_le_right d vi b;
    FStar.Math.Lemmas.lemma_mult_le_right d (vi + 1) b;
    (* forall r<vi. r*d+d <= vi*d  (turns mv_loop_step_lemma's prefix-guard
       r*d+d<=vi*d into the plain r<vi guard needed by transfer_rmnv_forall) *)
    mv_prefix_le_forall_lemma d vi;
    (* forall r<b. r*d+d <= b*d  (row-fit bounds for transfer/extend) *)
    row_lt_b_bound_forall_lemma d b;
    (* transfer rmnv from sx_pre to sx_post for every r < vi *)
    transfer_rmnv_forall bd d vi sx sx_pre sx_post eps inv_d;
    (* lift the vi-th row's rmnv first argument from sx_pre back to sx *)
    rmnv_lift_input_via_suffix #bd d vi bd eps inv_d sx sx_pre sx_post;
    (* extend the per-row forall from r<vi to r<vi+1 *)
    extend_row_norm_forall bd d b vi sx sx_post eps inv_d
#pop-options

(* ── l1_forward <-> seq bridges (row memcpy / reduce / map) ────────────

   The GPU buffers here use the identity row-major layout [l1_forward],
   whose index map is the identity, so its [to_seq]/[from_seq] flattening
   coincides with [chest1_to_seq]/[seq_to_chest1].  These pointwise
   identities are discharged by extensionality. *)

(* [to_seq] on [l1_forward] equals [chest1_to_seq]. *)
let lem_to_seq (#et:Type) (n:nat) (c : chest1 et n)
  : Lemma (to_seq (l1_forward n) c == chest1_to_seq c)
  = assert (Seq.equal (to_seq (l1_forward n) c) (chest1_to_seq c))

(* [chest1_to_seq] commutes with [chest_map]/[seq_map]. *)
let chest_map_to_seq (#et1 #et2 : Type) (#n : nat)
  (f : et1 -> et2) (c : chest1 et1 n)
  : Lemma (chest1_to_seq (chest_map f c) == KS.seq_map f (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (chest_map f c)) (KS.seq_map f (chest1_to_seq c)))

(* [chest1_to_seq] commutes with [to_real_chest]/[to_real_seq]. *)
let to_real_chest_to_seq (#et : Type0) {| scalar et, real_like et |} (#n : nat)
  (c : chest1 et n)
  : Lemma (chest1_to_seq (to_real_chest c) == to_real_seq (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (to_real_chest c)) (to_real_seq (chest1_to_seq c)))

(* Device-to-device offset blit at the chest level.  Bridges the raw
   [Kuiper.KB.Compat.Array.gpu_memcpy_device_to_device'] primitive (which works on the backing
   [core] larrays) through [tensor_concr]/[tensor_abs]/[tensor_abs'] and
   the [l1_forward] round-trip.  This is the direct replacement for the
   deleted [Array1.memcpy_device_to_device'] and keeps the SAME argument
   order (dst, dst_off, src, src_off, cnt).  [src] is preserved; [dst] is
   overwritten by the blit at offset [dst_off]. *)
inline_for_extraction noextract
fn t_memcpy_d2d'
  (#a:Type u#0) {| sized a |}
  (#dst_sz : erased nat)
  (dst : array1 a (l1_forward dst_sz))
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src : array1 a (l1_forward src_sz))
  (src_off : SZ.t)
  (cnt : SZ.t { SZ.v dst_off + SZ.v cnt <= dst_sz /\ SZ.v src_off + SZ.v cnt <= src_sz })
  (#f : perm)
  (#v : chest1 a src_sz)
  (#gv : chest1 a dst_sz)
  preserves cpu ** on gpu_loc (src |-> Frac f v)
  requires on gpu_loc (dst |-> gv)
  ensures exists* (s' : chest1 a dst_sz).
      on gpu_loc (dst |-> s') **
      pure (chest1_to_seq s' ==
            KS.seq_blit (chest1_to_seq gv) dst_off (chest1_to_seq v) src_off cnt)
{
  map_loc gpu_loc #(dst |-> gv) #(core dst |-> to_seq (l1_forward dst_sz) gv)
    fn _ { tensor_concr dst; };
  map_loc gpu_loc #(src |-> Frac f v) #(core src |-> Frac f (to_seq (l1_forward src_sz) v))
    fn _ { tensor_concr src; };
  Kuiper.KB.Compat.Array.gpu_memcpy_device_to_device' (core dst) dst_off (core src) src_off cnt;
  with s'seq. assert (on gpu_loc (core dst |-> s'seq));
  map_loc gpu_loc #(core src |-> Frac f (to_seq (l1_forward src_sz) v)) #(src |-> Frac f v)
    fn _ {
      tensor_abs (l1_forward src_sz) (core src);
      rewrite (from_array (l1_forward src_sz) (core src) |-> Frac f v)
           as (src |-> Frac f v);
    };
  map_loc gpu_loc #(core dst |-> s'seq) #(dst |-> from_seq (l1_forward dst_sz) s'seq)
    fn _ {
      tensor_abs' (l1_forward dst_sz) (core dst);
      rewrite (from_array (l1_forward dst_sz) (core dst) |-> from_seq (l1_forward dst_sz) s'seq)
           as (dst |-> from_seq (l1_forward dst_sz) s'seq);
    };
  lem_to_seq dst_sz gv;
  lem_to_seq src_sz v;
  lem_to_seq dst_sz (from_seq (l1_forward dst_sz) s'seq);
  ()
}

(* Per-row body: copy [x[r,:]] into scratch, sum-reduce for the mean,
   square scratch in place, sum-reduce for the second moment, refresh
   scratch from row, run an affine map (inv, -mean*inv), copy back. *)
(* Keep the two logical postconditions as separate pure resources.  The
   row-normalized predicate already binds all eight floating-point statistics,
   so duplicating those witnesses here only creates a very large SMT query. *)
#push-options "--z3rlimit 20 --fuel 1 --ifuel 2"
inline_for_extraction noextract
fn mean_var_norm_row
  (b : szp)
  (d : szp { d <= max_blocks * max_threads /\
             b * d <= max_blocks * max_threads })
  (rv_off : sz { rv_off + d <= b * d })
  (eps : f32)
  (inv_d : f32)
  (x : array1 f32 (l1_forward (b * d)) { is_global x })
  (scratch : array1 f32 (l1_forward d) { is_global scratch })
  (#sx : chest1 f32 (b * d))
  (#ss : chest1 f32 d)
  preserves cpu
  requires on gpu_loc (x |-> sx) ** on gpu_loc (scratch |-> ss) **
    pure (inv_d %~ (1.0R /. FStar.Real.of_int (SZ.v d))) **
    pure (row_mean_var_domain (chest1_to_seq sx) rv_off d eps)
  ensures
    (exists* (sx' : chest1 f32 (b * d)) (ss' : chest1 f32 d).
       on gpu_loc (x |-> sx') ** on gpu_loc (scratch |-> ss') **
       pure (row_mean_var_normalized (chest1_to_seq sx) (chest1_to_seq sx')
         rv_off d eps inv_d) **
       pure (exists (inv' neg_mean_inv' : f32).
         chest1_to_seq sx' == KS.seq_blit (chest1_to_seq sx) rv_off
           (affine_result #f32 inv' neg_mean_inv' #d
             (Seq.slice (chest1_to_seq sx) rv_off (SZ.v rv_off + SZ.v d)))
           0 d))
{
  (* Copy the row x[rv_off .. rv_off+d) into scratch[0 .. d). *)
  t_memcpy_d2d' scratch 0sz x rv_off d;
  with ss1. assert (on gpu_loc (scratch |-> reveal ss1));
  (* [seq_blit ss 0 sx rv_off d] fully overwrites the length-d scratch,
     so it equals [slice sx rv_off (rv_off+d)] -- the row. *)
  let row_g : erased (lseq f32 d) =
    hide (Seq.slice (chest1_to_seq (reveal sx)) rv_off (SZ.v rv_off + SZ.v d));
  Seq.lemma_eq_intro
    (KS.seq_blit (chest1_to_seq (reveal ss)) 0 (chest1_to_seq (reveal sx)) rv_off d)
    (reveal row_g);
  assert pure (chest1_to_seq (reveal ss1) == reveal row_g);

  (* Real-valued view of the (preserved) row, shared by both reductions.
     [ss1 %~ vr] fires via the [to_real_chest] SMTPat. *)
  let vr : chest1 real d = hide (to_real_chest (reveal ss1));
  assert pure (reveal ss1 %~ reveal vr);

  (* Pass 1: sum reduce → sum1 (device tree-reduce, identity pre-map). *)
  let sum1 = HRed.reduce #f32 id id 1024sz d scratch #ss1 vr;
  (* Bridge: chest_map id vr flattens (via seq_map id = id) to
     to_real_seq (chest1_to_seq ss1) = to_real_seq row_g. *)
  chest_map_to_seq (id #real) (reveal vr);
  seq_map_id_eq #real (chest1_to_seq (reveal vr));
  to_real_chest_to_seq (reveal ss1);
  assert pure (sum1 %~ rsum (to_real_seq (reveal row_g)));
  let mean = mul sum1 inv_d;

  (* Pass 2: sum reduce of squared values via [pre_map = square].  reduce
     preserves scratch, so it still holds ss1 here. *)
  sq_step_approx_forall #f32 ();
  let sum2 = HRed.reduce #f32 (square #f32) sq_step_r 1024sz d scratch #ss1 vr;
  (* Bridge: chest_map sq_step_r vr flattens to seq_map sq_step_r (to_real_seq row_g)
     = frobenius_sumsq_r (to_real_seq row_g). *)
  chest_map_to_seq sq_step_r (reveal vr);
  assert pure (sum2 %~ frobenius_sumsq_r (to_real_seq (reveal row_g)));
  let m2 = mul sum2 inv_d;
  let var = sub m2 (mul mean mean);
  let var_eps = add var eps;
  let inv = rsqrt var_eps;
  let neg_mean_inv = sub zero (mul mean inv);

  (* Pass 3: scratch still contains ss1; affine map, copy back. *)
  Map.map_gpu (affine_step inv neg_mean_inv) d scratch;
  t_memcpy_d2d' x rv_off scratch 0sz d;
  with vfinal. assert (on gpu_loc (x |-> reveal vfinal));
  (* Bridge the scaled scratch (chest_map (affine_step ..) ss1) back to the
     seq-level [affine_result inv neg_mean_inv row_g]. *)
  chest_map_to_seq (affine_step inv neg_mean_inv) (reveal ss1);
  assert pure (chest1_to_seq (reveal vfinal) == KS.seq_blit (chest1_to_seq (reveal sx)) rv_off
    (affine_result #f32 inv neg_mean_inv #d (reveal row_g))
    0 d);
  assert pure (sum1 %~ rsum (to_real_seq (reveal row_g)));
  assert pure (sum2 %~ frobenius_sumsq_r (to_real_seq (reveal row_g)));
  assert pure (mean == mul sum1 inv_d);
  assert pure (m2 == mul sum2 inv_d);
  assert pure (var == sub m2 (mul mean mean));
  assert pure (var_eps == add var eps);
  assert pure (inv == rsqrt var_eps);
  assert pure (neg_mean_inv == sub zero (mul mean inv));
  assert pure (Seq.length (chest1_to_seq (reveal vfinal)) == b * d);
  assert pure (rv_off + d <= b * d);
  (* [vfinal] is the blit of [affine_result] into [sx] at [rv_off]; the slice
     over the blitted region recovers [affine_result].  [blit_slice_inside]
     unfolds [seq_blit] for SMT (the pointwise [lemma_eq_intro] requires that
     unfolding, which no longer happens automatically after the merge). *)
  blit_slice_inside (chest1_to_seq (reveal sx))
    (affine_result #f32 inv neg_mean_inv #d (reveal row_g))
    rv_off 0 d;
  Seq.lemma_eq_intro
    (Seq.slice (affine_result #f32 inv neg_mean_inv #d (reveal row_g))
               0 d)
    (affine_result #f32 inv neg_mean_inv #d (reveal row_g));
  assert pure (Seq.slice (chest1_to_seq (reveal vfinal)) rv_off (SZ.v rv_off + SZ.v d) ==
               affine_result #f32 inv neg_mean_inv #d (reveal row_g));
  row_mean_var_normalized_intro #(b * d) d
    (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal vfinal))
    rv_off eps inv_d sum1 sum2 mean m2 var var_eps inv neg_mean_inv;
  assert pure (row_mean_var_normalized (chest1_to_seq (reveal sx))
    (chest1_to_seq (reveal vfinal)) rv_off d eps inv_d);
  assert pure (exists (inv_w neg_mean_inv_w : f32).
    chest1_to_seq (reveal vfinal) == KS.seq_blit (chest1_to_seq (reveal sx)) rv_off
      (affine_result #f32 inv_w neg_mean_inv_w #d
        (Seq.slice (chest1_to_seq (reveal sx)) rv_off
          (SZ.v rv_off + SZ.v d)))
      0 d);
  ()
}
#pop-options


inline_for_extraction noextract
(*: split each invariant-maintenance obligation into a
   separate Z3 query.  Without this, Z3 4.13 crashes (LP-solver assertion
   violation) when SizeT arithmetic (SZ.add) and a complex forall containing
   row_mean_var_normalized appear in the same combined query.
   --z3refresh --z3seed 2: match the green L2Norm.l2norm loop. *)
#push-options "--z3rlimit 300 --z3refresh --z3seed 2"
fn mean_var_norm
  (b : szp)
  (d : szp { d <= max_blocks * max_threads /\
             SZ.fits (b * d) /\
             b * d <= max_blocks * max_threads })
  (eps : f32)
  (inv_d : f32)
  (x : array1 f32 (l1_forward (b * d)) { is_global x })
  (#sx : chest1 f32 (b * d))
  preserves cpu
  requires on gpu_loc (x |-> sx) **
    pure (inv_d %~ (1.0R /. FStar.Real.of_int (SZ.v d))) **
    pure (mean_var_domain b d eps (chest1_to_seq sx))
  ensures
    (exists* (sx' : chest1 f32 (b * d)).
       on gpu_loc (x |-> sx') **
       pure (mean_var_float_post b d eps inv_d (chest1_to_seq sx) (chest1_to_seq sx')))
{
  let scratch = alloc0 #f32 d (l1_forward d);
  let mut idx = 0sz;
  (* Establish ∀r < b. r*d+d <= b*d BEFORE the loop so the fact
     persists in both the loop-body Z3 context (for off-refinement
     subtype checks) and the post-loop context (for mean_var_post).
     Proved once in a clean F* context via lemma_mult_le_right;
     thereafter Z3 needs only forall instantiation + SZ.mul axiom. *)
  row_lt_b_bound_forall_lemma d b;
  while (let i = !idx; SZ.(i <^ b))
    invariant
      exists* (vi : sz) (sx' : chest1 f32 (b * d)) (ss' : chest1 f32 d).
        idx |-> vi **
        on gpu_loc (x |-> sx') **
        on gpu_loc (scratch |-> ss') **
        cpu **
        pure (SZ.v vi <= SZ.v b /\
              (forall (r : nat). r < SZ.v vi ==>
                 row_mean_var_normalized (chest1_to_seq sx) (chest1_to_seq sx') (r * SZ.v d) d eps inv_d) /\
              Seq.slice (chest1_to_seq sx') (vi * d) (b * d) ==
                Seq.slice (chest1_to_seq sx) (vi * d) (b * d))
    decreases (SZ.v b - SZ.v !idx)
  {
    let i = !idx;
    let off : sz = SZ.(i *^ d);
    with sx_pre. assert (on gpu_loc (x |-> reveal sx_pre));
    assert pure (SZ.v off == SZ.v i * SZ.v d);
    row_mean_var_domain_via_suffix #(b*d) d i (b*d) eps
      (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx_pre));
    mean_var_norm_row b d off eps inv_d x scratch;
    with sx_post. assert (on gpu_loc (x |-> reveal sx_post));
    (* Ground the nonlinear per-row fit bound [forall r<b. r*d+d <= b*d] so
       the requires-[forall]s of the step lemma (whose [row_mean_var_normalized
       sx sx_pre (r*d) d] applications each carry the [(r*d)+d <= bd] subtype
       refinement) are well-typed without Z3 re-deriving the nonlinear bound
       under the quantifier in this large loop-body context. *)
    row_lt_b_bound_forall_lemma d b;
    (* One pure lemma re-establishes the whole loop invariant in a clean SMT
       context (exactly as the green L2Norm.l2norm loop calls
       l2_loop_invariant_step).  Its requires map onto the loop invariant
       (per-row rmnv forall + suffix-slice equality, over sx_pre = sx' here)
       and mean_var_norm_row's ensures (rmnv sx_pre sx_post at row i + the
       affine_result seq_blit characterisation).  The lemma's flat length is
       the mathematical product [b * d], exactly the type index of the
       [chest1_to_seq _] arguments. *)
    mv_loop_invariant_step (b * d) b d (chest1_to_seq (reveal sx))
      (chest1_to_seq (reveal sx_pre)) (chest1_to_seq (reveal sx_post)) i eps inv_d;
    idx := SZ.(!idx +^ 1sz);
  };
  free scratch;
  ()
}
#pop-options

let mvn_inv_d_approx (d:szp)
  : Lemma (mvn_inv_d #f32 d %~ (1.0R /. FStar.Real.of_int (SZ.v d)))
  = let d64 : Int64.t = FStar.Int.Cast.uint64_to_int64
      (FStar.SizeT.sizet_to_uint64 d) in
    assert (Int64.v d64 == SZ.v d);
    of_int_approx #f32 d64;
    div_approx (one #f32) (of_int #f32 d64)
      1.0R (FStar.Real.of_int (SZ.v d))

let mean_var_float_post_to_real
  (b:nat) (d:pos) (eps inv_d:f32)
  (sx sx':Seq.lseq f32 (b*d))
  : Lemma
      (requires mean_var_float_post b d eps inv_d sx sx')
      (ensures mean_var_post b d eps sx sx')
  = ()

(* Public entry point: compute the per-row reciprocal [mvn_inv_d d]
   inside the verification boundary (extracts to 1.0f / (float)d), then
   delegate to [mean_var_norm].  The heavy proof above treats [inv_d]
   abstractly, so this constant computation does not affect its cost. *)
fn mean_var_norm_fw
  (b : szp)
  (d : szp { d <= max_blocks * max_threads /\
             SZ.fits (b * d) /\
             b * d <= max_blocks * max_threads })
  (eps : f32)
  (x : array1 f32 (l1_forward (b * d)) { is_global x })
  (#sx : chest1 f32 (b * d))
  preserves cpu
  requires on gpu_loc (x |-> sx) **
    pure (mean_var_domain b d eps (chest1_to_seq sx))
  ensures
    (exists* (sx' : chest1 f32 (b * d)).
       on gpu_loc (x |-> sx') **
       pure (mean_var_post b d eps (chest1_to_seq sx) (chest1_to_seq sx')))
{
  let inv_d : f32 = mvn_inv_d d;
  mvn_inv_d_approx d;
  mean_var_norm b d eps inv_d x;
  with sx'. assert (on gpu_loc (x |-> reveal sx'));
  mean_var_float_post_to_real b d eps inv_d
    (chest1_to_seq (reveal sx)) (chest1_to_seq (reveal sx'));
}

let mean_var_norm_fw_f32 : mean_var_norm_fw_ty f32 = mean_var_norm_fw
