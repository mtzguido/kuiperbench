module Kuiper.Tensor.Layout.BCMPages

(* BCM-pages layout: view a (B, C, H, W) row-major buffer as a 2-D matrix
   (B*HW, C) where row r = batch b=r/HW at spatial position hw_idx=r%HW,
   and column ci = channel index.
   Physical offset: (r/HW)*C*HW + ci*HW + (r%HW). *)

open Kuiper
open Kuiper.Injection
open Kuiper.Shape
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Layout
module T = Kuiper.Tensor

(* The ghost layout (operates on nat indices). *)
[@@erasable]
val l2_bcm_pages (b : nat) (hw : nat{hw > 0}) (c : nat)
  : full_tlayout ((b * hw) @| c @| INil)

(* Expose .ulen so ctlayout can prove SZ.fits. *)
val l2_bcm_pages_ulen (b : nat) (hw : nat{hw > 0}) (c : nat)
  : Lemma ((l2_bcm_pages b hw c).ulen = (b * hw) * c)
          [SMTPat (l2_bcm_pages b hw c)]

(* Expose .imap.f so ctlayout can prove the cimap equation. *)
val l2_bcm_pages_imap_f (b : nat) (hw : nat{hw > 0}) (c : nat)
  (idx : abs ((b * hw) @| c @| INil))
  : Lemma (
      let (r, (ci, ())) = idx in
      (l2_bcm_pages b hw c).imap.f idx == (r / hw) * c * hw + ci * hw + (r % hw))
    [SMTPat ((l2_bcm_pages b hw c).imap.f idx)]

let bcm_offset_bound
  (b hw c q ci rem : nat)
  : Lemma
      (requires hw > 0 /\ q < b /\ ci < c /\ rem < hw)
      (ensures q * (c * hw) + ci * hw + rem < b * (hw * c))
  = ()

(* Concrete ctlayout instance for use in Pulse kernels. *)
#push-options "--z3rlimit 400 --fuel 3 --ifuel 3"
inline_for_extraction noextract
instance c_l2_bcm_pages
  (b  : erased nat)
  (hw : SZ.t{SZ.v hw > 0})
  (c  : SZ.t)
  (#_ : squash (SZ.fits (b * SZ.v hw)))
  (#_ : squash (SZ.fits (SZ.v hw * SZ.v c)))
  (#_ : squash (SZ.fits (b * (SZ.v hw * SZ.v c))))
  : T.ctlayout (l2_bcm_pages b hw c)
  = {
      ulen_fits = ();
      all_fit   = ();
      cimap = fun ((r, (ci, ())) : Kuiper.Shape.conc ((b * SZ.v hw) @| SZ.v c @| INil)) ->
        let q   = r /^ hw in
        let rem = r %^ hw in
        let ch  = c *^ hw in
        let qch = q *^ ch in
        let cih = ci *^ hw in
        let s1  = qch +^ cih in
        bcm_offset_bound b hw c q ci rem;
        SZ.fits_ok (SZ.v s1 + SZ.v rem);
        s1 +^ rem
    }
#pop-options
