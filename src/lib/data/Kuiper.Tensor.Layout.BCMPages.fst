module Kuiper.Tensor.Layout.BCMPages

open Kuiper
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg

#push-options "--z3rlimit 400 --fuel 3 --ifuel 3"
let l2_bcm_pages (b : nat) (hw : nat{hw > 0}) (c : nat)
  : full_tlayout ((b * hw) @| c @| INil)
  = let f : abs ((b * hw) @| c @| INil) -> GTot (natlt (sizeof ((b * hw) @| c @| INil))) =
      fun ((r, (ci, ())) : abs ((b * hw) @| c @| INil)) ->
        FStar.Math.Lemmas.euclidean_div_axiom r hw;
        let q   : natlt b  = r / hw in
        let rem : natlt hw = r % hw in
        (* Guide Z3: ci*hw + rem < c*hw, then q*(c*hw) + (ci*hw+rem) < b*(c*hw) *)
        assert (ci * hw + rem < c * hw);
        assert (q * c * hw + ci * hw + rem < (b * hw) * c);
        q * c * hw + ci * hw + rem
    in
    let is_inj : x : abs _ -> y : abs _{f x == f y} -> squash (x == y) =
      fun (x : abs _) (y : abs _{f x == f y}) ->
        let (r1, (ci1, ())) = x in
        let (r2, (ci2, ())) = y in
        FStar.Math.Lemmas.euclidean_div_axiom r1 hw;
        FStar.Math.Lemmas.euclidean_div_axiom r2 hw;
        let q1   : natlt b  = r1 / hw in
        let rem1 : natlt hw = r1 % hw in
        let q2   : natlt b  = r2 / hw in
        let rem2 : natlt hw = r2 % hw in
        (* f x == f y means: q1*c*hw + ci1*hw + rem1 = q2*c*hw + ci2*hw + rem2 *)
        assert (q1 * c * hw + ci1 * hw + rem1 = q2 * c * hw + ci2 * hw + rem2);
        (* Step 1: deduce rem1 = rem2 via mod hw *)
        FStar.Math.Lemmas.lemma_mod_plus rem1 (q1 * c + ci1) hw;
        FStar.Math.Lemmas.lemma_mod_plus rem2 (q2 * c + ci2) hw;
        FStar.Math.Lemmas.distributivity_add_left (q1 * c) ci1 hw;
        FStar.Math.Lemmas.distributivity_add_left (q2 * c) ci2 hw;
        FStar.Math.Lemmas.small_mod rem1 hw;
        FStar.Math.Lemmas.small_mod rem2 hw;
        assert (rem1 = rem2);
        (* Step 2: cancel rem, divide by hw: q1*c + ci1 = q2*c + ci2 *)
        assert ((q1*c + ci1) * hw = (q2*c + ci2) * hw);
        assert (q1*c + ci1 = q2*c + ci2);
        (* Step 3: deduce ci1 = ci2 via mod c *)
        FStar.Math.Lemmas.lemma_mod_plus (q1*c + ci1) 1 c;
        FStar.Math.Lemmas.lemma_mod_plus (q2*c + ci2) 1 c;
        assert (ci1 = ci2);
        (* Step 4: q1 = q2 *)
        assert (q1*c = q2*c);
        assert (q1 = q2);
        (* Step 5: r1 = r2 *)
        assert (r1 = q1 * hw + rem1);
        assert (r2 = q2 * hw + rem2)
    in
    pack (mk_injection f is_inj)
#pop-options

#push-options "--z3rlimit 200 --fuel 3 --ifuel 3"
let l2_bcm_pages_ulen (b : nat) (hw : nat{hw > 0}) (c : nat)
  : Lemma ((l2_bcm_pages b hw c).ulen = (b * hw) * c)
  = ()
#pop-options

#push-options "--z3rlimit 100 --fuel 3 --ifuel 3"
let l2_bcm_pages_imap_f (b : nat) (hw : nat{hw > 0}) (c : nat)
  (idx : abs ((b * hw) @| c @| INil))
  : Lemma (
      let (r, (ci, ())) = idx in
      (l2_bcm_pages b hw c).imap.f idx == (r / hw) * c * hw + ci * hw + (r % hw))
  = let (r, (ci, ())) = idx in
    FStar.Math.Lemmas.euclidean_div_axiom r hw
#pop-options
