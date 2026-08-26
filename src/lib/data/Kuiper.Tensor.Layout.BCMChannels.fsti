module Kuiper.Tensor.Layout.BCMChannels

(* BCM-channels layout: view a (N, C, H, W) row-major buffer as a 2-D
   matrix (C, N*HW) where row c is the entirety of channel c (all batch
   and spatial positions), and column k = n*HW + (h*W + w) flattens
   (n, hw_idx).
   Physical offset: (k/HW)*C*HW + c*HW + (k%HW)
                  = n*C*HW + c*HW + spatial. *)

open Kuiper
open Kuiper.Injection
open Kuiper.Shape
module SZ = Kuiper.SizeT
open Kuiper.Tensor.Layout
module T = Kuiper.Tensor

(* The ghost layout (operates on nat indices). *)
[@@erasable]
val l2_bcm_channels (n : nat) (c : nat) (hw : nat{hw > 0})
  : full_tlayout (c @| (n * hw) @| INil)

(* Expose .ulen so ctlayout can prove SZ.fits. *)
val l2_bcm_channels_ulen (n : nat) (c : nat) (hw : nat{hw > 0})
  : Lemma ((l2_bcm_channels n c hw).ulen = n * c * hw)
          [SMTPat (l2_bcm_channels n c hw)]

(* Expose .imap.f so ctlayout can prove the cimap equation. *)
val l2_bcm_channels_imap_f (n : nat) (c : nat) (hw : nat{hw > 0})
  (idx : abs (c @| (n * hw) @| INil))
  : Lemma (
      let (ci, (k, ())) = idx in
      (l2_bcm_channels n c hw).imap.f idx == (k / hw) * c * hw + ci * hw + (k % hw))
    [SMTPat ((l2_bcm_channels n c hw).imap.f idx)]

(* All the [SZ.fits] obligations for the physical-offset arithmetic of the
   [cimap] below, bundled into one abstract lemma.  Keeping it abstract (a
   [val] here, proved once in the [.fst]) means the [inline_for_extraction]
   instance re-checks cheaply when inlined into client [.fst] modules, where
   the high-rlimit context of this interface is not in scope. *)
val cimap_fits (n : nat) (c hw : pos) (ci k : nat)
  : Lemma (requires ci < c /\ k < n * hw /\
                    SZ.fits (n * hw) /\ SZ.fits (hw * c) /\ SZ.fits (n * (hw * c)))
          (ensures (let q = k / hw in
                    let rem = k % hw in
                    SZ.fits (c * hw) /\
                    SZ.fits (q * (c * hw)) /\
                    SZ.fits (ci * hw) /\
                    SZ.fits (q * (c * hw) + ci * hw) /\
                    SZ.fits (q * (c * hw) + ci * hw + rem)))

(* Concrete ctlayout instance for use in Pulse kernels. *)
#push-options "--z3rlimit 400 --fuel 3 --ifuel 3"
inline_for_extraction noextract
instance c_l2_bcm_channels
  (n  : erased nat)
  (c  : SZ.t)
  (hw : SZ.t{SZ.v hw > 0})
  (#_ : squash (SZ.fits (n * SZ.v hw)))
  (#_ : squash (SZ.fits (SZ.v hw * SZ.v c)))
  (#_ : squash (SZ.fits (n * (SZ.v hw * SZ.v c))))
  : T.ctlayout (l2_bcm_channels n (SZ.v c) (SZ.v hw))
  = {
      ulen_fits = ();
      all_fit   = ();
      cimap = fun ((ci, (k, ())) : Kuiper.Shape.conc (SZ.v c @| (n * SZ.v hw) @| INil)) ->
        cimap_fits n (SZ.v c) (SZ.v hw) (SZ.v ci) (SZ.v k);
        let q   = k /^ hw in
        let rem = k %^ hw in
        let ch  = c *^ hw in
        let qch = q *^ ch in
        let cih = ci *^ hw in
        let s1  = qch +^ cih in
        s1 +^ rem
    }
#pop-options
