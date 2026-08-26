module Kuiper.Tensor.Layout.BCMChannels

open Kuiper
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT

#push-options "--z3rlimit 400 --fuel 3 --ifuel 3"
let l2_bcm_channels (n : nat) (c : nat) (hw : nat{hw > 0})
  : full_tlayout (c @| (n * hw) @| INil)
  = let f : abs (c @| (n * hw) @| INil) -> GTot (natlt (sizeof (c @| (n * hw) @| INil))) =
      fun ((ci, (k, ())) : abs (c @| (n * hw) @| INil)) ->
        FStar.Math.Lemmas.euclidean_div_axiom k hw;
        let q   : natlt n  = k / hw in
        let rem : natlt hw = k % hw in
        (* Guide Z3: q*C*hw + ci*hw + rem < n*C*hw *)
        assert (ci * hw + rem < c * hw);
        assert (q * c * hw + ci * hw + rem < n * c * hw);
        assert (n * c * hw == c * (n * hw));
        q * c * hw + ci * hw + rem
    in
    let is_inj : x : abs _ -> y : abs _{f x == f y} -> squash (x == y) =
      fun (x : abs _) (y : abs _{f x == f y}) ->
        let (ci1, (k1, ())) = x in
        let (ci2, (k2, ())) = y in
        FStar.Math.Lemmas.euclidean_div_axiom k1 hw;
        FStar.Math.Lemmas.euclidean_div_axiom k2 hw;
        let q1   : natlt n  = k1 / hw in
        let rem1 : natlt hw = k1 % hw in
        let q2   : natlt n  = k2 / hw in
        let rem2 : natlt hw = k2 % hw in
        assert (q1 * c * hw + ci1 * hw + rem1 = q2 * c * hw + ci2 * hw + rem2);
        (* Step 1: rem1 = rem2 via mod hw *)
        FStar.Math.Lemmas.lemma_mod_plus rem1 (q1 * c + ci1) hw;
        FStar.Math.Lemmas.lemma_mod_plus rem2 (q2 * c + ci2) hw;
        FStar.Math.Lemmas.small_mod rem1 hw;
        FStar.Math.Lemmas.small_mod rem2 hw;
        assert (rem1 + (q1 * c + ci1) * hw == q1 * c * hw + ci1 * hw + rem1);
        assert (rem2 + (q2 * c + ci2) * hw == q2 * c * hw + ci2 * hw + rem2);
        assert (rem1 = rem2);
        (* Step 2: q1*c + ci1 = q2*c + ci2 *)
        assert ((q1*c + ci1) * hw = (q2*c + ci2) * hw);
        assert (q1*c + ci1 = q2*c + ci2);
        (* Step 3: ci1 = ci2 via mod c *)
        FStar.Math.Lemmas.lemma_mod_plus ci1 q1 c;
        FStar.Math.Lemmas.lemma_mod_plus ci2 q2 c;
        FStar.Math.Lemmas.small_mod ci1 c;
        FStar.Math.Lemmas.small_mod ci2 c;
        assert (ci1 = ci2);
        (* Step 4: q1 = q2 *)
        assert (q1*c = q2*c);
        assert (q1 = q2);
        (* Step 5: k1 = k2 *)
        assert (k1 = q1 * hw + rem1);
        assert (k2 = q2 * hw + rem2)
    in
    pack (mk_injection f is_inj)
#pop-options

#push-options "--z3rlimit 200 --fuel 3 --ifuel 3"
let l2_bcm_channels_ulen (n : nat) (c : nat) (hw : nat{hw > 0})
  : Lemma ((l2_bcm_channels n c hw).ulen = n * c * hw)
  = ()
#pop-options

#push-options "--z3rlimit 100 --fuel 3 --ifuel 3"
let l2_bcm_channels_imap_f (n : nat) (c : nat) (hw : nat{hw > 0})
  (idx : abs (c @| (n * hw) @| INil))
  : Lemma (
      let (ci, (k, ())) = idx in
      (l2_bcm_channels n c hw).imap.f idx == (k / hw) * c * hw + ci * hw + (k % hw))
  = let (ci, (k, ())) = idx in
    FStar.Math.Lemmas.euclidean_div_axiom k hw
#pop-options

#push-options "--z3rlimit 200 --fuel 2 --ifuel 2"
let cimap_fits (n : nat) (c hw : pos) (ci k : nat)
  : Lemma (requires ci < c /\ k < n * hw /\
                    SZ.fits (n * hw) /\ SZ.fits (hw * c) /\ SZ.fits (n * (hw * c)))
          (ensures (let q = k / hw in
                    let rem = k % hw in
                    SZ.fits (c * hw) /\
                    SZ.fits (q * (c * hw)) /\
                    SZ.fits (ci * hw) /\
                    SZ.fits (q * (c * hw) + ci * hw) /\
                    SZ.fits (q * (c * hw) + ci * hw + rem)))
  = let q   = k / hw in
    let rem = k % hw in
    (* q < n: q*hw <= k < n*hw, cancel hw. *)
    FStar.Math.Lemmas.division_propriety k hw;
    FStar.Math.Lemmas.multiplication_order_lemma q n hw;
    assert (q < n);
    (* c*hw == hw*c, so it fits by the [hw*c] hypothesis. *)
    FStar.Math.Lemmas.swap_mul c hw;
    (* q*(c*hw) <= (n-1)*(c*hw); ci*hw <= (c-1)*hw. *)
    FStar.Math.Lemmas.lemma_mult_le_right (c * hw) q (n - 1);
    FStar.Math.Lemmas.lemma_mult_le_right hw ci (c - 1);
    (* Whole offset < n*(c*hw) == n*(hw*c), which fits. *)
    assert (q * (c * hw) + ci * hw + rem < n * (c * hw));
    FStar.Math.Lemmas.swap_mul c hw
#pop-options
