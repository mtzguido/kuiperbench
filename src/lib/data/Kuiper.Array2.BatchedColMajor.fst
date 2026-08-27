module Kuiper.Array2.BatchedColMajor

(* A bespoke "flattened batched-column-major" 2-index Array2 layout.

   Motivation (KernelBench L1 #42 MaxPool2D, separable 2-D pool).
   Pass 1 of a separable 2-D pool reduces a row-major (bc, H, W)
   buffer over the inner W axis, producing a row-major
   (bc, H, W_out) intermediate.  Pass 2 must reduce over H, which is
   *not* the inner axis of that buffer.  Rather than physically
   transposing, we view the SAME (bc, H, W_out) row-major bytes as a
   flat 2-index matrix of shape (rows = bc*W_out, cols = H) whose
   element (R, j) is the byte the 3-index view places at
   (b = R/W_out, i = j, w = R%W_out):

     offset(R, j) = (R / wq) * (inner * wq) + j * wq + (R % wq)

   with [wq = W_out] and [inner = H].  [windowreduce] then reduces
   over the inner (H) axis with no data movement.

   This [imap] is a valid injection but is NOT a nested-major
   ([major_on]) layout, so it is built by hand here:
     - [flat_bcm_off]      : the physical offset (a nat);
     - [flat_bcm_bound]    : it lands in [natlt (rows*cols)];
     - [flat_bcm_inj]      : injectivity (mixed-radix uniqueness);
     - [flat_bcm]          : the [Array2.layout] (full);
     - [c_flat_bcm]        : the extractable [ctlayout] instance. *)

open Kuiper
open Kuiper.Injection
open Kuiper.Shape
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas
open FStar.SizeT { ( /^ ), ( %^ ), ( +^ ), ( -^ ), ( *^ ) }

(* ── Physical offset ─────────────────────────────────────────────── *)

let flat_bcm_off (bc : nat) (wq : pos) (inner : nat)
  (rr : natlt (bc * wq)) (j : natlt inner)
  : nat
  = (rr / wq) * (inner * wq) + j * wq + (rr % wq)

(* a*n < b*n with n>0 implies a < b. *)
let cancel_lt (a b : nat) (n : pos)
  : Lemma (requires a * n < b * n) (ensures a < b)
  = if a < b then ()
    else ML.lemma_mult_le_right n b a   (* b<=a ==> b*n<=a*n, contradiction *)

(* The offset lands strictly below the buffer size (bc*wq)*inner. *)
let flat_bcm_bound (bc : nat) (wq : pos) (inner : nat)
  (rr : natlt (bc * wq)) (j : natlt inner)
  : Lemma (flat_bcm_off bc wq inner rr j < (bc * wq) * inner)
  = let b = rr / wq in
    let w = rr % wq in
    ML.lemma_div_mod rr wq;                 (* rr == wq*b + w, 0<=w<wq *)
    let n = inner * wq in
    (* b < bc : from b*wq <= rr < bc*wq *)
    assert (b * wq <= rr);
    cancel_lt b bc wq;
    assert (b < bc);
    (* j*wq + w < inner*wq = n *)
    ML.lemma_mult_le_right wq j (inner - 1);
    ML.distributivity_sub_left inner 1 wq;  (* (inner-1)*wq == inner*wq - wq *)
    assert (j * wq + w < n);
    (* b*n + n <= bc*n, and b*n + (j*wq+w) < b*n + n *)
    ML.lemma_mult_le_right n (b + 1) bc;
    ML.distributivity_add_left b 1 n;
    ML.swap_mul bc n;
    assert (b * n + (j * wq + w) < bc * n)

let flat_bcm_off_b (bc : nat) (wq : pos) (inner : nat)
  (rr : natlt (bc * wq)) (j : natlt inner)
  : natlt ((bc * wq) * inner)
  = flat_bcm_bound bc wq inner rr j;
    flat_bcm_off bc wq inner rr j

(* ── Injectivity (mixed-radix uniqueness) ────────────────────────── *)

(* If a = q1*n + r1 = q2*n + r2 with 0<=r1,r2<n then q1=q2 and r1=r2. *)
let divmod_unique (n : pos) (q1 r1 q2 r2 : nat)
  : Lemma (requires r1 < n /\ r2 < n /\ q1 * n + r1 == q2 * n + r2)
          (ensures q1 == q2 /\ r1 == r2)
  = ML.lemma_div_plus r1 q1 n;     (* (r1 + q1*n)/n = r1/n + q1 *)
    ML.lemma_div_plus r2 q2 n;
    ML.small_div r1 n;             (* r1/n = 0 *)
    ML.small_div r2 n;
    ML.lemma_mod_plus r1 q1 n;     (* (r1 + q1*n)%n = r1%n *)
    ML.lemma_mod_plus r2 q2 n;
    ML.small_mod r1 n;
    ML.small_mod r2 n;
    assert (q1 * n + r1 == r1 + q1 * n);
    assert (q2 * n + r2 == r2 + q2 * n)

let flat_bcm_inj (bc : nat) (wq : pos) (inner : nat)
  (x : abs ((bc * wq) @| inner @| INil))
  (y : abs ((bc * wq) @| inner @| INil) {
        flat_bcm_off_b bc wq inner x._1 x._2._1
          == flat_bcm_off_b bc wq inner y._1 y._2._1 })
  : squash (x == y)
  = let (r1, (j1, ())) = x in
    let (r2, (j2, ())) = y in
    let n = inner * wq in
    ML.lemma_div_mod r1 wq;
    ML.lemma_div_mod r2 wq;
    let b1 = r1 / wq in let w1 = r1 % wq in
    let b2 = r2 / wq in let w2 = r2 % wq in
    (* offsets equal: b_i*n + (j_i*wq + w_i) *)
    ML.swap_mul (inner) wq;
    (* j_i*wq + w_i < n *)
    ML.lemma_mult_le_right wq j1 (inner - 1);
    ML.lemma_mult_le_right wq j2 (inner - 1);
    ML.distributivity_sub_left inner 1 wq;
    assert (j1 * wq + w1 < n);
    assert (j2 * wq + w2 < n);
    assert (b1 * n + (j1 * wq + w1) == b2 * n + (j2 * wq + w2));
    divmod_unique n b1 (j1 * wq + w1) b2 (j2 * wq + w2);
    (* now b1==b2 and j1*wq+w1 == j2*wq+w2 *)
    divmod_unique wq j1 w1 j2 w2;
    (* j1==j2, w1==w2, b1==b2 => r1==r2 *)
    assert (r1 == b1 * wq + w1);
    assert (r2 == b2 * wq + w2)

(* ── The layout ──────────────────────────────────────────────────── *)

let flat_bcm_layout_f (bc : nat) (wq : pos) (inner : nat)
  : layout_f_for ((bc * wq) @| inner @| INil)
  = size_layout_2 (bc * wq) inner;
    mk_injection
      #(abs ((bc * wq) @| inner @| INil)) #(natlt (sizeof ((bc * wq) @| inner @| INil)))
      (fun (idx : abs ((bc * wq) @| inner @| INil)) ->
        let (rr, (j, ())) = idx in
        flat_bcm_off_b bc wq inner rr j)
      (fun x y -> flat_bcm_inj bc wq inner x y)

let flat_bcm (bc : nat) (wq : pos) (inner : nat)
  : layout2 (bc * wq) inner
  = pack (flat_bcm_layout_f bc wq inner)

let flat_bcm_full (bc : nat) (wq : pos) (inner : nat)
  : Lemma (is_full (flat_bcm bc wq inner))
  = ()

(* The defining equation of the layout's imap (exposed for the
   composition proof). *)
let flat_bcm_imap (bc : nat) (wq : pos) (inner : nat)
  (rr : natlt (bc * wq)) (j : natlt inner)
  : Lemma ((flat_bcm bc wq inner).imap.f (rr, (j, ()))
           == (rr / wq) * (inner * wq) + j * wq + (rr % wq))
  = ()

(* ── Concrete (extractable) ctlayout instance ───────────────────── *)

(* The extractable index function, factored out so the [fits] side
   conditions of the SZ arithmetic can be discharged with named
   lemmas (instance bodies cannot carry assertions). *)
inline_for_extraction noextract
let flat_bcm_cimap
  (bc : erased nat) (wq : SZ.t{SZ.v wq > 0}) (inner : SZ.t)
  (sq_rows  : squash (SZ.fits (bc * SZ.v wq)))
  (sq_plane : squash (SZ.fits (SZ.v inner * SZ.v wq)))
  (sq_tot   : squash (SZ.fits ((bc * SZ.v wq) * SZ.v inner)))
  (idx : conc ((bc * SZ.v wq) @| inner @| INil))
  : (y : SZ.t { SZ.v y == (flat_bcm bc wq inner).imap.f (up idx) })
  = let (rr, (j, ())) = idx in
    let b = SZ.v rr / SZ.v wq in
    let n = SZ.v inner * SZ.v wq in
    cancel_lt b bc wq;                 (* b < bc *)
    ML.lemma_mult_le_right n b (bc - 1);      (* b*n <= (bc-1)*n *)
    ML.distributivity_sub_left bc 1 n;        (* (bc-1)*n == bc*n - n *)
    ML.swap_mul bc n;                         (* bc*n == n*bc == (bc*wq)*inner via... *)
    ML.paren_mul_right bc inner wq;
    ML.swap_mul inner wq;
    flat_bcm_bound bc wq inner rr j;
    (rr /^ wq) *^ (inner *^ wq) +^ j *^ wq +^ (rr %^ wq)

inline_for_extraction noextract
instance c_flat_bcm
  (bc : erased nat) (wq : SZ.t{SZ.v wq > 0})
  (inner : SZ.t)
  (#_ : squash (SZ.fits (bc * SZ.v wq)))
  (#_ : squash (SZ.fits (SZ.v inner * SZ.v wq)))
  (#_ : squash (SZ.fits ((bc * SZ.v wq) * SZ.v inner)))
  : ctlayout (flat_bcm bc wq inner)
  = {
      ulen_fits = ();
      all_fit = ();
      cimap = flat_bcm_cimap bc wq inner () () ();
    }
