module Kuiper.Kernel.HReduce

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.GhostReference { read as gread, write as gwrite, alloc as galloc }
open Kuiper.Tensor
open Kuiper.Chest1.Helpers
open Kuiper.Bijection { ( =~ ), bij_sym }
// Re-open after Kuiper.Tensor so the seq-level `@!`/`seq![..]`/`@+` notations
// shadow the shape-indexing `@!` pulled in via Kuiper.Shape.
open Kuiper.Seq.Common

module SZ = Kuiper.SizeT
module RPM = Kuiper.Barrier.RPM
module B = Kuiper.Barrier

(* Bijection between the abstract 1-D tensor index [(k, ())] and a plain
   [natlt len], used to (un)reindex a forevery over tensor cells. *)
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

(* Same bijection but with a refined [nat] domain, matching the binder produced
   by unfolding [array1_pts_to_slice r 0 nth]. *)
let nat_abs_bij (nth : nat) : ((k:nat{0 <= k /\ k < nth}) =~ abs (nth @| INil)) =
  {
    ff = (fun k -> ((k, ()) <: abs (nth @| INil)));
    gg = (fun (i, ()) -> i);
  }

(* [chest1_to_seq] commutes with mapping; bridges the chest1 interface (in the
   .fsti) to the seq-based numeric proof below. *)
let chest_map_to_seq_map (#et1 #et2 : Type) (#n : nat)
  (f : et1 -> et2) (c : chest1 et1 n)
  : Lemma (chest1_to_seq (chest_map f c) == seq_map f (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (chest_map f c)) (seq_map f (chest1_to_seq c)))

let acc1_chest1_append_left (#et : Type) (#n #m : nat)
  (s1 : chest1 et n) (s2 : chest1 et m) (i : natlt n)
  : Lemma (acc1 (chest1_append s1 s2) i == acc1 s1 i)
          [SMTPat (acc1 (chest1_append s1 s2) i)]
  = ()

let acc1_chest1_append_right (#et : Type) (#n #m : nat)
  (s1 : chest1 et n) (s2 : chest1 et m) (i : natlt m)
  : Lemma (acc1 (chest1_append s1 s2) (n + i) == acc1 s2 i)
          [SMTPat (acc1 (chest1_append s1 s2) (n + i))]
  = ()

let acc1_chest1_append_boundary (#et : Type) (#n : nat) (#m : pos)
  (s1 : chest1 et n) (s2 : chest1 et m)
  : Lemma (acc1 (chest1_append s1 s2) n == acc1 s2 0)
          [SMTPat (acc1 (chest1_append s1 s2) n)]
  = ()

(* Plain ownership of a slice of a 1-D tensor. *)
let array1_pts_to_slice
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey]i
   [@@@mkey]j : nat{i <= j /\ j <= sz})
  (s : chest1 et (j - i))
  : slprop
  = forall+ (k : nat{i <= k /\ k < j}).
      Cell r ((k, ()) <: abs (sz @| INil)) |-> (s `acc1` (k - i))

#push-options "--z3rlimit 80"
ghost
fn array1_slice_concat
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i j k : nat{i <= j /\ j <= k /\ k <= sz})
  (#s1 : chest1 et (j - i))
  (#s2 : chest1 et (k - j))
  requires
    array1_pts_to_slice r i j s1 **
    array1_pts_to_slice r j k s2
  ensures
    array1_pts_to_slice r i k (chest1_append s1 s2)
{
  unfold array1_pts_to_slice r i j s1;
  unfold array1_pts_to_slice r j k s2;

  let s = chest1_append s1 s2;

  (* Rewrite each side to use s *)
  forevery_ext
    (fun (x:nat{i <= x /\ x < j}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s1 (x - i)))
    (fun (x:nat{i <= x /\ x < j}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s (x - i)));
  forevery_ext
    (fun (x:nat{j <= x /\ x < k}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s2 (x - j)))
    (fun (x:nat{j <= x /\ x < k}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s (x - i)));

  (* Join *)
  forevery_refine_join' #nat
    (fun (x:nat) -> i <= x /\ x < j)
    (fun (x:nat) -> j <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s (x - i)));

  (* Simplify *)
  forevery_refine_ext' #nat
    #(fun (x:nat) -> (i <= x /\ x < j) \/ (j <= x /\ x < k))
    (fun (x:nat) -> i <= x /\ x < k)
    (fun (x:nat{(i <= x /\ x < j) \/ (j <= x /\ x < k)}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 s (x - i)));

  // FIXME: Terrible that this is needed. But we have a length of (j-i)+(k-j) which
  // does not unify with k-i.
  forevery_ext
    _
    (fun (x : nat{i <= x /\ x < k}) ->
      tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 #_ #(k-i) s (x - i)));

  fold array1_pts_to_slice r i k s;
}
#pop-options

inline_for_extraction noextract
fn array1_read_from_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : layout1 len) {| ctlayout l |}
  (r : array1 et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : chest1 et (j - i))
  preserves
    array1_pts_to_slice r i j s
  returns
    v : et
  ensures
    pure (v == acc1 s (idx - i))
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract #(x:nat{i <= x /\ x < j}) idx _;
  let v = tensor_read_cell r ((idx <: szlt len), ());
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s;
  v
}

inline_for_extraction noextract
fn array1_write_to_slice
  (#et : Type0)
  (#len : erased nat)
  (#l : layout1 len) {| ctlayout l |}
  (r : array1 et l)
  (#i #j : erased nat{i <= j /\ j <= len})
  (idx : sz{i <= idx /\ idx < j})
  (#s : chest1 et (j - i))
  (v : et)
  requires
    array1_pts_to_slice r i j s
  ensures
    array1_pts_to_slice r i j (upd1 s (idx - i) v)
{
  unfold array1_pts_to_slice r i j s;
  forevery_extract' #(x:nat{i <= x /\ x < j}) idx _;
  tensor_write_cell r ((idx <: szlt len), ()) v;
  let s' : chest1 et (j - i) = upd1 s (idx - i) v;
  Pulse.Lib.Forall.elim_forall
    (fun (x:nat{i <= x /\ x < j}) ->
      tensor_pts_to_cell r ((x <: natlt len), ()) (acc1 s' (x - i)));
  Pulse.Lib.Trade.elim_trade _ _;
  fold array1_pts_to_slice r i j s';
  rewrite each s' as upd1 s (idx - i) v;
  ()
}

(* Build a length-one slice from a single owned cell. *)
ghost
fn slice_singleton
  (#et : Type0)
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i : nat{i < sz})
  (v : et)
  (s : chest1 et 1)
  requires
    tensor_pts_to_cell r ((i <: natlt sz), ()) v **
    pure (v == acc1 s 0)
  ensures
    array1_pts_to_slice r i (i+1) s
{
  forevery_singleton_intro'
    #(x:nat{i <= x /\ x < i + 1})
    (fun x -> tensor_pts_to_cell r ((x <: natlt sz), ()) v)
    i;
  forevery_ext
    (fun (x:nat{i <= x /\ x < i + 1}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) v)
    (fun (x:nat{i <= x /\ x < i + 1}) -> tensor_pts_to_cell r ((x <: natlt sz), ()) (acc1 #_ #(i+1-i) s (x - i)));
  fold array1_pts_to_slice r i (i+1) s;
}

(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
unfold
let array1_pts_to_slice_sum_inner
  (#et:Type0) {| scalar et, real_like et |}
  (#sz : nat)
  (#l : layout1 sz)
  (r : array1 et l)
  (i j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  (s : chest1 et (j - i))
  : slprop
  = array1_pts_to_slice r i j s **
    pure ((acc1 s 0) %~ rsum (Seq.slice rr i j))

let array1_pts_to_slice_sum
  (#et:Type0) {| scalar et, real_like et |}
  (#sz : nat)
  (#l : layout1 sz)
  ([@@@mkey] r : array1 et l)
  ([@@@mkey] i : nat)
  (j : nat{i < j /\ j <= sz})
  (rr : lseq real sz)
  : slprop
  = exists* s. array1_pts_to_slice_sum_inner r i j rr s

// Barrier

unfold let barrier_matrix
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp)
  (#l : layout1 nth)
  (r : array1 et l)
  (vr : lseq real nth)
  (it : nat)
  (from to : natlt nth)
: slprop
=
  if_ (from = to + pow2 it)
      (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
           (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr))

ghost
fn mk_barrier_pre
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp)
  (#l : layout1 nth)
  (r : array1 et l)
  (vr : lseq real nth)
  (tid : natlt nth)
  (it: natlt 31)
  requires
    if_ (not (div_pow2 (it + 1) tid) && div_pow2 it tid)
      (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr)
  ensures
    forall+ (i:natlt nth). barrier_matrix nth r vr it tid i
{
  open FStar.SizeT;
  if (tid >= pow2 it) {
    forevery_if_intro #(natlt nth) (tid - pow2 it) (fun i ->
      if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
        (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr));
    forevery_ext
      (fun (i:natlt nth) ->
        if_ (op_Equals #(natlt nth) i (tid - pow2 it))
          (if_ (not (div_pow2 (it + 1) tid) && (div_pow2 it tid))
            (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr)))
      (fun (i:natlt nth) -> barrier_matrix nth r vr it tid i);
  } else {
    assert pure (pow2 it > tid);
    assert pure (tid % pow2 it == tid);
    if_elim_false _;
    forevery_emp_intro (natlt nth);
    forevery_ext
      (fun (i:natlt nth) -> emp)
      (fun (i:natlt nth) -> barrier_matrix nth r vr it tid i);
  }
}

// RO permission to a, thread 0 also owns output ref, full ownership of own cell
// in shmem array
unfold
let kpre
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena)
  (a : array1 et l)
  (va : chest1 et lena)
  (vr : chest1 real lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  (tid : natlt nth)
  : slprop
  = a |-> Frac (1 /. nth) va **
    if_ (op_Equals #nat tid 0) (live out) **
    exists* (v : et).
      tensor_pts_to_cell (from_array (l1_forward nth) shmem._1) (tid, ()) v

// Same RO permission to a, 1st thread has full ownership of shmem plus of the
// output reference.  No need to specify the contents of the shmem array, it
// will disappear.
unfold
let kpost
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena)
  (a : array1 et l)
  (va : chest1 et lena)
  (vr : chest1 real lena)
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  (tid : natlt nth)
  : slprop
  = a |-> Frac (1 /. nth) va **
    if_ (op_Equals #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))
    )

inline_for_extraction
fn iteration
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp { SZ.v nth <= max_threads })
  (#l : layout1 nth) {| Kuiper.Tensor.ctlayout l |}
  (r : array1 et l)
  (vr : erased (lseq real nth))
  (tid : szlt nth)
  (it: szlt 31)
  preserves gpu
  preserves thread_id nth tid
  preserves mbarrier_tok nth (barrier_matrix nth r vr)
  requires B.barrier_state it
  requires if_ (div_pow2 it tid) (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr)
  ensures  B.barrier_state (it + 1)
  ensures  if_ (div_pow2 (it+1) tid) (array1_pts_to_slice_sum r tid (min (tid + pow2 (it + 1)) nth) vr)
{
  case_split (div_pow2 (it + 1) tid)
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr));
  if_flatten #(div_pow2 (it + 1) tid);
  if_flatten #(not (div_pow2 (it + 1) tid));

  div_pow2_lemma it (it + 1) tid;
  rewrite (if_ (div_pow2 (it + 1) tid && div_pow2 it tid)
            (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr))
      as (if_ (div_pow2 (it + 1) tid)
            (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vr));

  mk_barrier_pre nth r vr tid it;
  fold RPM.row (barrier_matrix nth r vr) it tid;
  mbarrier_wait ();
  unfold RPM.col (barrier_matrix nth r vr) it tid;

  // combine (div_pow2 (it + 1) tid) (array1_pts_to_slice_sum r tid (min (tid + pow2 it) nth) vv) _;

  let nextid = FStar.SizeT.(tid +^ spow2 it);

  (* We do not use end_ in extracted code, so we can use a nat and erase it
  so there are no traces in the extracted C. *)
  let end_ : erased nat = hide (min (tid + 2 * pow2 it) nth);

  if (nextid <^ nth) {
    forevery_ext
      (fun (from: natlt nth) ->
        if_ (op_Equals #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr)))
      (fun (from: natlt nth) ->
        if_ (op_Equals #(natlt nth) from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
            (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr)));
    forevery_if_elim #(natlt nth)
      (tid + pow2 it)
      (fun (from: natlt nth) -> if_ (not (div_pow2 (it + 1) from) && (div_pow2 it from))
         (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr));

    let b = sdiv_pow2 (it +^ 1sz) tid;

    rewrite each (div_pow2 (it + 1) tid) as b;

    div_pow2_lemma_2 it tid;
    combine
      b
      (array1_pts_to_slice_sum r nextid (min (tid + pow2 it + pow2 it) nth) vr)
      _;

    if b {
      assert (pure (div_pow2 (SZ.v it + 1) tid));
      if_elim_true _;

      (**)unfold (array1_pts_to_slice_sum r nextid end_ vr);
      (**)with s_right. assert
        (array1_pts_to_slice_sum_inner r nextid end_ vr s_right);
      (**)unfold (array1_pts_to_slice_sum r tid nextid vr);
      (**)with s_left. assert
        (array1_pts_to_slice_sum_inner r tid nextid vr s_left);
      (**)array1_slice_concat #et #nth r tid nextid end_;

      let s1 = array1_read_from_slice r tid;
      (**)assert pure (SZ.v tid - SZ.v tid == 0);
      (**)acc1_chest1_append_left s_left s_right 0;
      (**)assert (pure (s1 `approximates` rsum (Seq.slice vr tid nextid)));

      let s2 = array1_read_from_slice r nextid;
      (**)acc1_chest1_append_boundary s_left s_right;
      (**)assert (pure (s2 `approximates` rsum (Seq.slice vr nextid end_)));

      let s = add s1 s2;
      (**)lem_append_slice vr tid nextid end_;
      (**)seq_approximates_append s1 s2 (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure ((s1 `add` s2) `approximates` rsum (Seq.append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_))));
      (**)rsum_append (Seq.slice vr tid nextid) (Seq.slice vr nextid end_);
      (**)assert (pure (s `approximates` rsum (Seq.slice vr tid end_)));

      // gpu_array_write r tid s;
      array1_write_to_slice r tid s;

      (**)with seq. assert (array1_pts_to_slice r tid end_ seq);
      (**)fold (array1_pts_to_slice_sum r tid end_ vr);
      (**)if_intro_true (array1_pts_to_slice_sum r tid end_ vr);
      // Step below optional right now, but good practice?
      (**)rewrite
      (**)  if_ true
      (**)      (array1_pts_to_slice_sum r tid (reveal end_) vr)
      (**)as
      (**)  if_ (div_pow2 (SZ.v it + 1) tid)
      (**)      (array1_pts_to_slice_sum r tid (reveal end_) vr);
    } else {
      (* no-op *)
      if_elim_false _;
      if_intro_false (array1_pts_to_slice_sum r tid end_ vr);
    }
  } else {
    forevery_map
      (fun (from: natlt nth) ->
        if_ (op_Equals #int from (tid + pow2 it))
          (if_ (not (div_pow2 (it + 1) from) && div_pow2 it from)
            (array1_pts_to_slice_sum r from (min (from + pow2 it) nth) vr)))
      (fun from -> emp)
      fn from {
        if_rewrite_bool (from = tid + pow2 it) false _;
        if_elim_false _;
      };
    forevery_emp_elim _;
  }
}

(* Number of barrier calls in the reduction loop: smallest k s.t. pow2 k >= nth *)
let hreduce_barrier_count (nth : pos) : GTot nat = log2 (2 * nth - 1)

(* If pow2 k <= n < pow2 (k+1), then log2 n = k. *)
let rec log2_range (n:pos) (k:nat)
  : Lemma (requires pow2 k <= n /\ n < pow2 (k+1))
          (ensures log2 n == k)
          (decreases k)
= if k = 0 then ()
  else begin
    FStar.Math.Lemmas.lemma_div_le (pow2 k) n 2;
    log2_range (n/2) (k-1)
  end

(* The smallest k with pow2 k >= nth equals log2 (2*nth - 1). *)
let log2_hreduce (nth:pos) (it:nat)
  : Lemma (requires pow2 it >= nth /\ (it == 0 \/ pow2 (it - 1) < nth))
          (ensures it == log2 (2 * nth - 1))
= if it = 0 then ()
  else log2_range (2 * nth - 1) it

(* ---- Helpers for strided_sum_is_sum ---- *)

private let rsum_snoc_ (s : seq real) (x : real)
  : Lemma (rsum (Seq.snoc s x) == rsum s +. x)
  = assert (Seq.equal (Seq.snoc s x) (Seq.append s (Seq.create 1 x)));
    rsum_append s (Seq.create 1 x)

#push-options "--z3rlimit 20"
private let stride_len_snoc_ (n : nat) (nth : pos) (off : natlt nth)
  : Lemma (ensures (
      let old_len = (n - off + nth - 1) / nth in
      let new_len = (n + 1 - off + nth - 1) / nth in
      if n % nth = off then new_len == old_len + 1
      else new_len == old_len))
  = let m = n - off + nth - 1 in
    FStar.Math.Lemmas.euclidean_division_definition m nth;
    FStar.Math.Lemmas.euclidean_division_definition (m+1) nth;
    FStar.Math.Lemmas.euclidean_division_definition n nth;
    FStar.Math.Lemmas.modulo_addition_lemma (n % nth + nth - 1 - off) nth (n / nth);
    assert (m % nth == (n % nth + nth - 1 - off) % nth);
    if n % nth <= off then begin
      FStar.Math.Lemmas.small_mod (n % nth + nth - 1 - off) nth
    end else begin
      FStar.Math.Lemmas.modulo_addition_lemma (n % nth - 1 - off) nth 1;
      FStar.Math.Lemmas.small_mod (n % nth - 1 - off) nth
    end
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 30"
private let seq_stride_snoc_ (s : seq real) (x : real) (nth : pos) (off : natlt nth)
  : Lemma (ensures (
      let s' = Seq.snoc s x in
      let n = Seq.length s in
      if n % nth = off
      then Seq.equal (seq_stride s' nth off) (Seq.snoc (seq_stride s nth off) x)
      else Seq.equal (seq_stride s' nth off) (seq_stride s nth off)))
  = let n = Seq.length s in
    let s' = Seq.snoc s x in
    stride_len_snoc_ n nth off;
    let old_stride = seq_stride s nth off in
    let new_stride = seq_stride s' nth off in
    if n % nth = off then begin
      let old_sl = seq_stride_length s nth off in
      let new_sl = seq_stride_length s' nth off in
      assert (new_sl == old_sl + 1);
      let goal = Seq.snoc old_stride x in
      let aux (i : nat{i < new_sl}) : Lemma (Seq.index new_stride i == Seq.index goal i)
        = if i < old_sl then
            assert (off + i * nth < n)
          else begin
            assert (i == old_sl);
            assert (off + old_sl * nth == n)
          end
      in
      Classical.forall_intro aux;
      assert (Seq.equal new_stride goal)
    end else begin
      let old_sl = seq_stride_length s nth off in
      let aux (i : nat{i < old_sl}) : Lemma (Seq.index new_stride i == Seq.index old_stride i)
        = assert ((off + i * nth) % nth == off);
          assert (off + i * nth <> n);
          assert (off + i * nth < n)
      in
      Classical.forall_intro aux;
      assert (Seq.equal new_stride old_stride)
    end
#pop-options

#push-options "--z3rlimit 10"
private let rsum_seq_take_next_ (s : seq real) (n : nat{n < Seq.length s})
  : Lemma (rsum (seq_take n s) +. (s @! n) == rsum (seq_take (n + 1) s))
  = assert (Seq.equal (seq_take (n + 1) s) (Seq.snoc (seq_take n s) (s @! n)));
    rsum_snoc_ (seq_take n s) (s @! n)

private let rsum_singleton_ (x : real)
  : Lemma (rsum (Seq.create 1 x) == x)
  = let SCons hd tl = view_seq (Seq.create 1 x) in
    assert (Seq.equal tl (Seq.empty #real))

private let rsum_upd_ (s : seq real) (k : nat{k < Seq.length s}) (v : real)
  : Lemma (rsum (Seq.upd s k v) == rsum s +. (v -. (s @! k)))
  = let s1 = Seq.slice s 0 k in
    let s2 = Seq.slice s (k+1) (Seq.length s) in
    assert (Seq.equal s (s1 @+ Seq.create 1 (s @! k) @+ s2));
    assert (Seq.equal (Seq.upd s k v) (s1 @+ Seq.create 1 v @+ s2));
    rsum_append s1 (Seq.create 1 (s @! k) @+ s2);
    rsum_append (Seq.create 1 (s @! k)) s2;
    rsum_append s1 (Seq.create 1 v @+ s2);
    rsum_append (Seq.create 1 v) s2;
    rsum_singleton_ (s @! k);
    rsum_singleton_ v
#pop-options

private let rec rsum_zeros_ (n : nat)
  : Lemma (ensures rsum (Seq.init_ghost n (fun _ -> 0.0R)) == 0.0R)
          (decreases n)
  = if n = 0 then ()
    else begin
      let s = Seq.init_ghost n (fun (_:nat{_ < n}) -> 0.0R) in
      let SCons hd tl = view_seq s in
      assert (hd == 0.0R);
      assert (Seq.equal tl (Seq.init_ghost (n-1) (fun (_:nat{_ < n-1}) -> 0.0R)));
      rsum_zeros_ (n-1)
    end

#push-options "--z3rlimit 20"
private let rec strided_sum_is_sum_core_ (s : seq real) (nth : pos)
  : Lemma (ensures rsum (Seq.init_ghost nth (fun tid -> rsum (seq_stride s nth tid))) == rsum s)
          (decreases Seq.length s)
  = if Seq.length s = 0 then begin
      assert (Seq.equal s (Seq.empty #real));
      let aux (tid : natlt nth) : Lemma (rsum (seq_stride s nth tid) == 0.0R)
        = assert (seq_stride_length s nth tid == 0);
          assert (Seq.equal (seq_stride s nth tid) (Seq.empty #real))
      in
      Classical.forall_intro aux;
      let ig = Seq.init_ghost nth (fun tid -> rsum (seq_stride s nth tid)) in
      rsum_zeros_ nth;
      let z = Seq.init_ghost nth (fun _ -> 0.0R) in
      assert (forall (tid:natlt nth). ig @! tid == 0.0R);
      assert (forall (tid:natlt nth). z @! tid == 0.0R);
      Seq.lemma_eq_elim ig z;
      assert (rsum ig == 0.0R);
      assert (rsum s == 0.0R)
    end else begin
      let s', last = Seq.un_snoc s in
      let n = Seq.length s' in
      let off : natlt nth = n % nth in

      strided_sum_is_sum_core_ s' nth;
      assert (Seq.equal s (Seq.snoc s' last));
      rsum_snoc_ s' last;

      let f  (tid : natlt nth) : GTot real = rsum (seq_stride s  nth tid) in
      let f' (tid : natlt nth) : GTot real = rsum (seq_stride s' nth tid) in

      let aux (tid : natlt nth) : Lemma (
        if tid = off then f tid == f' tid +. last
        else f tid == f' tid)
        = seq_stride_snoc_ s' last nth tid;
          if tid = off then
            rsum_snoc_ (seq_stride s' nth tid) last
          else ()
      in
      Classical.forall_intro aux;

      let ig  = Seq.init_ghost nth (fun tid -> rsum (seq_stride s nth tid)) in
      let ig' = Seq.init_ghost nth (fun tid -> rsum (seq_stride s' nth tid)) in

      let upd_ig' = Seq.upd ig' off (rsum (seq_stride s' nth off) +. last) in
      let eq_aux (i : natlt nth) : Lemma (ig @! i == upd_ig' @! i)
        = if i = off then
            assert (f i == f' i +. last)
          else
            assert (f i == f' i)
      in
      Classical.forall_intro eq_aux;
      assert (Seq.equal ig upd_ig');

      rsum_upd_ ig' off (rsum (seq_stride s' nth off) +. last);
      assert (ig' @! off == rsum (seq_stride s' nth off));
      rsum_singleton_ (rsum (seq_stride s' nth off) +. last);
      rsum_singleton_ (rsum (seq_stride s' nth off))
    end
#pop-options

(* ---- End helpers ---- *)

let vr_partial (pre_map : real -> real) (vr : seq real) (nth : nat) : GTot (seq real) =
  Seq.init_ghost nth (fun tid -> rsum (seq_stride (seq_map pre_map vr) nth tid))

let strided_sum_is_sum (pre_map : real -> real) (vr : seq real) (nth : pos)
  : Lemma (ensures rsum (vr_partial pre_map vr nth) == rsum (seq_map pre_map vr))
  = strided_sum_is_sum_core_ (seq_map pre_map vr) nth

let lemma_first_past
  (len off : nat) (stride : pos)
  (i : nat)
  : Lemma (requires i % stride == off /\ i >= len /\ i < len + stride)
          (ensures  i == off + ((len - off - 1 + stride) / stride) * stride)
  = ()

#push-options "--z3rlimit 60"
inline_for_extraction noextract
fn sum_stride_map
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (lena : sz)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l)
  (stride : szp)
  (off : szlt stride)
  (#va : chest1 et lena)
  (vr : chest1 real lena)
  (#f : perm)
  preserves
    gpu ** a |-> Frac f va ** pure (va %~ vr) ** pure (SZ.fits (lena + stride))
  returns
    res : et
  ensures
    pure (res %~ rsum (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off))
{
  let mut acc : et = zero;
  let mut idx : sz = off;
  let gidx = galloc #nat 0;

  while (!idx <^ lena)
    invariant
      live acc ** live gidx **
      live idx **
      pure (SZ.v !idx == gread gidx * stride + off /\
            off <= SZ.v !idx /\ SZ.v !idx < lena + stride /\
            gread gidx <= seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off /\
            // gread gidx == (SZ.v !idx - SZ.v off) / SZ.v stride /\ // superfluous
            !acc %~ rsum (seq_take (gread gidx) (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off))) **
      emp
    decreases (lena + stride - !idx)
  {
    assert pure (gread gidx < seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off);

    (* Read from input array (fractional permission) *)
    let vidx = !idx;
    let v = tensor_read a ((vidx <: szlt lena), ());
    let v' = pre_map v;
    (**)assert (pure (v == acc1 va (SZ.v !idx)));
    (**)assert (pure (v %~ (chest1_to_seq vr @! SZ.v !idx)));
    (**)assert (pure (v' %~ (seq_map pre_map_r (chest1_to_seq vr) @! SZ.v !idx)));

    assert pure (!acc %~ rsum (seq_take (gread gidx) (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off)));
    assert pure (v %~ (chest1_to_seq vr @! !idx));
    assert pure (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off @! gread gidx == (seq_map pre_map_r (chest1_to_seq vr)) @! (off + gread gidx * stride));
    assert pure (off + gread gidx * stride == SZ.v !idx);
    rsum_seq_take_next_ (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off) (gread gidx);

    let vgidx = gread gidx;
    assert (pure (SZ.v !idx                  == vgidx    * stride + off));
    Math.Lemmas.distributivity_add_left vgidx 1 stride;
    assert (pure ((vgidx + 1) * stride + off == ((vgidx * stride) + (1 * stride)) + off)); // Sad.
    assert (pure (SZ.v !idx + stride == (vgidx + 1) * stride + off));

    Math.Lemmas.add_div_mod_1 (SZ.v !idx) stride;

    acc := !acc `add` v';
    idx := !idx +^ stride;
    gwrite gidx (gread gidx + 1);

    assert pure (SZ.v !idx == gread gidx * stride + off);
    ()
  };

  assert pure (SZ.v !idx == gread gidx * stride + off);
  Math.Lemmas.lemma_mod_plus off (gread gidx) stride;
  Math.Lemmas.small_mod off stride;
  assert pure ((off + stride * gread gidx) % stride == off);
  assert pure ((gread gidx * stride + off) % stride == off);
  assert pure (!idx % stride == off);
  lemma_first_past lena off stride (SZ.v !idx);
  assert (pure (SZ.v !idx == off + ((lena - off - 1 + stride) / stride) * stride));

  assert pure (gread gidx <= seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off);
  Math.Lemmas.cancel_mul_div (gread gidx) stride;
  (* A calc proof would be much nicer. *)
  assert pure (gread gidx == (!idx - off) / stride);
  assert pure (gread gidx == ((off + ((lena - off - 1 + stride) / stride) * stride) - off) / stride);
  assert pure (gread gidx == (((lena - off - 1 + stride) / stride) * stride) / stride);
  assert pure (gread gidx == (lena - off - 1 + stride) / stride);
  assert pure (lena - off - 1 + stride == lena - off + stride - 1);
  assert pure (gread gidx == (lena - off + stride - 1) / stride);
  assert pure (gread gidx == seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off);
  assert pure (seq_take (seq_stride_length (seq_map pre_map_r (chest1_to_seq vr)) stride off) (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off) == seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off);
  assert pure (!acc %~ rsum (seq_stride (seq_map pre_map_r (chest1_to_seq vr)) stride off));

  drop_ (gidx |-> _);

  !acc
}
#pop-options

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kf
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l)
  (va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : szlt 1sz)
  (tid : szlt nth)
  ()
  preserves gpu
  requires
    pure (c_shmems_inv shmem) **
    kpre pre_map pre_map_r nth lena a va vr out shmem bid tid **
    thread_id nth tid **
    block_id 1sz bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1) (vr_partial pre_map_r (chest1_to_seq vr) nth)) **
    B.barrier_state 0
  ensures
    kpost pre_map pre_map_r nth lena a va vr out shmem bid tid **
    thread_id nth tid **
    block_id 1sz bid **
    mbarrier_tok nth (barrier_matrix nth (from_array (l1_forward nth) shmem._1) (vr_partial pre_map_r (chest1_to_seq vr) nth)) **
    B.barrier_state (hreduce_barrier_count nth)
{
  let (gsa, _) = shmem;

  let sa = from_array (l1_forward nth) gsa;
  rewrite each from_array (l1_forward nth) gsa as sa;

  (* Compute partial sum and write to shmem *)
  (**)chest1_to_seq_approx va vr;
  let psum : et = sum_stride_map pre_map pre_map_r lena a nth tid vr;
  tensor_write_cell sa (tid, ()) psum;

  (* Now do tree reduction on shmem *)
  let mut n : szlt 32 = 0sz;

  let psum_chest : chest1 et 1 = mk1 #et #1 (fun _ -> psum);
  slice_singleton sa tid psum psum_chest;

  (**)fold (array1_pts_to_slice_sum sa tid (tid + 1) (vr_partial pre_map_r (chest1_to_seq vr) nth));
  (**)if_intro_true' (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) (vr_partial pre_map_r (chest1_to_seq vr) nth));

  open FStar.SizeT;
  while (spow2 !n <^ nth)
    invariant
      live n **
      B.barrier_state !n **
      if_ (div_pow2 !n tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 !n) nth) (vr_partial pre_map_r (chest1_to_seq vr) nth)) **
      pure (v !n > 0 ==> pow2 (v !n - 1) < v nth)
    decreases (2 * nth - spow2 !n)
  {
    iteration nth sa (vr_partial pre_map_r (chest1_to_seq vr) nth) tid !n;
    n := !n +^ 1sz;
  };

  with it. assert (B.barrier_state it);

  // After loop exit: pow2 it >= nth, and tid < nth, so div_pow2 it tid <==> tid = 0
  FStar.Math.Lemmas.modulo_lemma tid (pow2 it);
  rewrite
    (if_ (div_pow2 it tid) (array1_pts_to_slice_sum sa tid (min (tid + pow2 it) nth) (vr_partial pre_map_r (chest1_to_seq vr) nth)))
  as
    (if_ (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial pre_map_r (chest1_to_seq vr) nth)));

  log2_hreduce (v nth) it;
  rewrite (B.barrier_state it) as (B.barrier_state (hreduce_barrier_count nth));

  (* Thread zero owns the result at the end, and writes it out. *)
  if (tid = 0sz) {
    if_elim_true' (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial pre_map_r (chest1_to_seq vr) nth));
    if_elim_true' (op_Equals #nat tid 0) (live out);
    unfold array1_pts_to_slice_sum sa 0 nth (vr_partial pre_map_r (chest1_to_seq vr) nth);
    (**)strided_sum_is_sum pre_map_r (chest1_to_seq vr) nth;
    (**)chest_map_to_seq_map pre_map_r vr;
    (**)assert (pure (Seq.equal (Seq.slice (vr_partial pre_map_r (chest1_to_seq vr) nth) 0 nth) (vr_partial pre_map_r (chest1_to_seq vr) nth)));
    gpu_write out (array1_read_from_slice sa 0sz);
    with ss. assert array1_pts_to_slice sa 0 nth ss;
    unfold array1_pts_to_slice sa;
    let css : chest1 et nth = hide (mk1 #et #nth (fun (k:natlt nth) -> acc1 ss k));
    (* Clean the index refinement [0<=k /\ k<nth] down to [k<nth] (= natlt nth),
       then reindex to the abstract tensor index and implode. *)
    forevery_refine_ext' #nat #(fun (k:nat) -> 0 <= k /\ k < nth) (fun (k:nat) -> k < nth) _;
    forevery_ext
      (fun (k:natlt nth) -> tensor_pts_to_cell sa ((k <: natlt nth), ()) (acc1 ss (k - 0)))
      (fun (k:natlt nth) -> tensor_pts_to_cell sa (abs_bij.gg k) (acc (reveal css) (abs_bij.gg k)));
    forevery_iso_back (abs_bij #nth)
      (fun (i : abs (nth @| INil)) -> tensor_pts_to_cell sa i (acc (reveal css) i));
    tensor_implode sa #1.0R #(reveal css);
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    if_intro_true' (op_Equals #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))
    )
  } else {
    (* Nop, convince Pulse. *)
    if_elim_false' (op_Equals #nat tid 0) (array1_pts_to_slice_sum sa 0 nth (vr_partial pre_map_r (chest1_to_seq vr) nth));
    if_elim_false' (op_Equals #nat tid 0) (live out);
    if_intro_false' (op_Equals #nat tid 0) (
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))
    );
    rewrite each sa as from_array (l1_forward nth) shmem._1;
    ();
  };
}
#pop-options

ghost
fn block_setup
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena)
  (a : array1 et l)
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  ()
  norewrite
  requires
    live_c_shmems shmem **
    (a |-> va ** live out)
  ensures
    (forall+ (i : natlt nth). kpre pre_map pre_map_r nth lena a va vr out shmem bid i) **
    emp
{
  unfold_live_c_shmems_cons shmem #_;
  unfold_live_c_shmems_nil shmem._2 #_;
  let gsa = shmem._1; rewrite each fst shmem as gsa;
  unfold live_c_shmem gsa;

  with vgsa. assert gsa |-> vgsa;
  gpu_pts_to_ref gsa;

  (* share input into nth fractional copies *)
  tensor_share_n a nth;

  (* tid 0 gets the ref *)
  forevery_if_intro #(natlt nth) 0 (fun _ -> live out);
  (* Sad.*)
  forevery_ext
    (fun tid -> if_ (op_Equals #(natlt nth) tid 0) (live out))
    (fun tid -> if_ (op_Equals #nat tid 0) (live out));

  forevery_zip (fun _ -> a |-> Frac (1 /. nth) va) _;

  (* View shmem array as a tensor and explode it into per-cell ownership. *)
  tensor_abs' (l1_forward nth) gsa;
  tensor_explode (from_array (l1_forward nth) gsa);
  forevery_iso abs_bij _;

  forevery_zip #(natlt nth)
    (fun tid -> a |-> Frac (1 /. nth) va ** if_ (op_Equals #nat tid 0) (live out))
    _;

  forevery_map
    #(natlt nth)
    (fun tid ->
      (a |-> Frac (1 /. nth) va **
       if_ (op_Equals #nat tid 0) (live out)) **
      Cell (from_array (l1_forward nth) gsa) (abs_bij.gg (tid <: natlt nth))
        |-> (acc (from_seq (l1_forward nth) vgsa) (abs_bij.gg (tid <: natlt nth)))
    )
    (fun (tid : natlt nth) -> kpre pre_map pre_map_r nth lena a va vr out shmem bid tid)
    fn tid {
      rewrite each gsa as shmem._1;
      ();
    };

  ()
}


ghost
fn block_teardown
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena)
  (a : array1 et l)
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  (shmem : c_shmems [SHArray et nth])
  (bid : natlt 1)
  ()
  norewrite
  requires
    (forall+ (i : natlt nth). kpost pre_map pre_map_r nth lena a va vr out shmem bid i) **
    emp
  ensures
    live_c_shmems shmem **
    (a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))))
{
  forevery_unzip _ _;

  tensor_gather_n a nth;

  (* Sad.*)
  forevery_ext #(natlt nth)
    (fun tid ->
      if_ (op_Equals #nat tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))))
    (fun tid ->
      if_ (op_Equals #(natlt nth) tid 0) (
        live (from_array (l1_forward nth) shmem._1) **
        exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))));

  forevery_if_elim #(natlt nth) 0 (fun tid ->
      live (from_array (l1_forward nth) shmem._1) **
      exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))
  );

  tensor_concr (from_array (l1_forward nth) shmem._1);
  rewrite each core (from_array (l1_forward nth) shmem._1) as shmem._1;

  fold_live_c_shmems_nil shmem._2 #_;
  with vgsa. assert shmem._1 |-> vgsa;
  fold_live_c_shmem shmem._1;
  fold_live_c_shmems_cons shmem #_;
}

ghost
fn setup
  (#et:Type0) {| scalar et, real_like et |}
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  ()
  norewrite
  requires
    a |-> va ** live out
  ensures
    (forall+ (bid : natlt 1). a |-> va ** live out) **
    emp
{
  forevery_singleton_intro #(natlt 1) (fun _bid -> a |-> va ** live out);
}

ghost
fn teardown
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  ()
  norewrite
  requires
    (forall+ (bid : natlt 1). a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))) **
    emp
  ensures
    a |-> va ** (exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr))))
{
  forevery_singleton_elim #(natlt 1) _;
}

inline_for_extraction noextract
let kernel
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz { SZ.fits (lena + nth) })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena { va %~ vr })
  (out : gpu_ref et)
  : kernel_desc
      (a |-> va ** live out)
      (a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr))))
  = {
    nblk = 1sz;
    nthr = nth;

    shmems_desc = [SHArray et nth];

    barrier_contract = (fun _bid shmem ->
      mbarrier_contract (barrier_matrix #et nth (from_array _ shmem._1) (vr_partial pre_map_r (chest1_to_seq vr) nth)));
    barrier_count    = (fun _bid    -> hreduce_barrier_count nth);
    barrier_ok       = (fun _bid shmem ->
      mbarrier_transform (barrier_matrix nth #(l1_forward nth) (from_array _ shmem._1) (vr_partial pre_map_r (chest1_to_seq vr) nth)));

    f = kf pre_map pre_map_r nth lena a va vr out;

    block_pre  = (fun bid -> a |-> va ** live out);
    block_post = (fun bid -> a |-> va ** exists* (v : et). out |-> v ** pure (v %~ rsum (chest1_to_seq (chest_map pre_map_r vr))));
    setup      = setup    nth lena a #va vr out;
    teardown   = teardown pre_map pre_map_r nth lena a #va vr out;

    block_frame    = (fun _shmem _bid -> emp);
    block_setup    = block_setup    pre_map pre_map_r nth lena a #va vr out;
    block_teardown = block_teardown pre_map pre_map_r nth lena a #va vr out;

    kpre =  kpre  pre_map pre_map_r nth lena a va vr out;
    kpost = kpost pre_map pre_map_r nth lena a va vr out;
    frame = emp;

    // FIXME: kpre and kpost mention a non-global array, but tc resolution tries
    // to apply the instance for global arrays anyway, and fails to prove the
    // refinement.
    kpre_sendable       = magic();
    kpost_sendable      = magic();
    block_post_sendable = solve;
    block_pre_sendable  = solve;
  }

inline_for_extraction noextract
fn reduce
  (#et:Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (nth : szp { nth <= max_threads })
  (lena : sz)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (vr : chest1 real lena)
  norewrite // sigh... spec in fsti is not purified
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (va %~ vr) **
    pure (SZ.fits (lena + nth))
  returns
    res : et
  ensures
    pure (res %~ rsum (chest1_to_seq (chest_map pre_map_r vr)))
{
  let out = Kuiper.Ref.gpu_alloc0 #et ();
  launch_sync (kernel pre_map pre_map_r nth lena a vr out);

  (* Bring back out result, free swap. *)
  let mut hout : et = zero #et;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  !hout;
}

(* ══════════════════════════════════════════════════════════════════════════
   reduce_batched: one thread per row, each row fully reduced serially.
   ══════════════════════════════════════════════════════════════════════════ *)

module EM     = Kuiper.EMatrix
(* ── Unfolding lemmas for row_reduce_partial (opaque to SMT) ─────────── *)

let row_reduce_partial_zero
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (#rows #cols : nat)
  (sx : EM.chest2 et rows cols)
  (r : natlt rows)
  : Lemma (row_reduce_partial pre_map sx r 0 == zero)
          [SMTPat (row_reduce_partial pre_map sx r 0)]
  = assert_norm (row_reduce_partial pre_map sx r 0 == zero)

#push-options "--fuel 2 --ifuel 1"
let row_reduce_partial_succ
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (#rows #cols : nat)
  (sx : EM.chest2 et rows cols)
  (r : natlt rows)
  (k : nat{k < cols})
  : Lemma (row_reduce_partial pre_map sx r (k + 1) ==
           row_reduce_partial pre_map sx r k `add` pre_map (acc2 sx r k))
          [SMTPat (row_reduce_partial pre_map sx r (k + 1))]
  = reveal_opaque (`%row_reduce_partial) (row_reduce_partial pre_map sx r (k + 1))
#pop-options

(* ── Per-thread predicates ─────────────────────────────────────────────── *)

unfold
let kpre_batched
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (rows : szp)
  (cols : szp)
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : EM.chest2 et rows cols)
  (sout : chest1 et rows)
  (r : natlt rows)
  : slprop
  = x |-> Frac (1.0R /. SZ.v rows) sx **
    Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout r

unfold
let kpost_batched
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (rows : szp)
  (cols : szp)
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (sx   : EM.chest2 et rows cols)
  (sout : chest1 et rows)
  (r : natlt rows)
  : slprop
  = x |-> Frac (1.0R /. SZ.v rows) sx **
    Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> row_reduce pre_map sx r

(* ── Per-thread kernel function ────────────────────────────────────────── *)

#push-options "--fuel 2 --ifuel 2 --z3rlimit 400"
inline_for_extraction noextract
fn kf_batched
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (rows : szp)
  (cols : szp)
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)              {| ctlayout lout |}
  (x      : array2 et lin)
  (output : array1 et lout)
  (#sx   : EM.chest2 et rows cols)
  (#sout : chest1 et rows)
  (gid : szlt rows)
  ()
  norewrite
  preserves gpu
  requires
    kpre_batched pre_map rows cols x output sx sout gid
  ensures
    kpost_batched pre_map rows cols x output sx sout gid
{
  unfold kpre_batched pre_map rows cols x output sx sout gid;

  let mut ci_ref : sz = 0sz;
  let mut acc_ref : et = zero;

  while (!ci_ref <^ cols)
    invariant exists* (ci_v : SZ.t) (acc_v : et).
      ci_ref |-> ci_v **
      acc_ref |-> acc_v **
      x |-> Frac (1.0R /. SZ.v rows) sx **
      Cell output (((SZ.v gid <: natlt rows), ()) <: abs (SZ.v rows @| INil)) |-> acc1 sout gid **
      pure (SZ.v ci_v <= SZ.v cols /\
            acc_v == row_reduce_partial pre_map sx gid ci_v)
    decreases (SZ.v cols - SZ.v !ci_ref)
  {
    let ci_v_raw = !ci_ref;
    let ci_v : szlt cols = ci_v_raw;
    let v = tensor_read x (cidx2 gid ci_v);
    let acc_v = !acc_ref;
    assert pure (
      row_reduce_partial pre_map sx gid (SZ.v ci_v + 1) ==
      row_reduce_partial pre_map sx gid ci_v `add` pre_map (acc2 sx gid ci_v));
    acc_ref := add acc_v (pre_map v);
    ci_ref := !ci_ref +^ 1sz;
  };

  with acc_v. assert acc_ref |-> acc_v;
  let final_acc = !acc_ref;
  tensor_write_cell output ((gid <: szlt rows), ()) final_acc;

  fold kpost_batched pre_map rows cols x output sx sout gid;
}
#pop-options

(* ── Ghost setup ────────────────────────────────────────────────────────── *)

ghost
fn setup_batched
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (#sx   : EM.chest2 et rows cols)
  (#sout : chest1 et rows)
  ()
  norewrite
  requires
    x |-> sx ** output |-> sout
  ensures
    (forall+ (r : natlt rows). kpre_batched pre_map rows cols x output sx sout r) **
    pure (SZ.fits (tlayout_ulen lout))
{
  (* Establish fits fact while output is in whole-array form. *)
  tensor_pts_to_ref output;

  (* Share x among rows threads. *)
  tensor_share_n x rows;

  (* Explode output into per-cell cells. *)
  tensor_explode output;
  forevery_iso (abs_bij #rows) _;

  (* Zip x-fracs and output-cells. *)
  forevery_zip #(natlt rows)
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (r : natlt rows) -> Cell output (abs_bij.gg r) |-> acc sout (abs_bij.gg r));

  (* Rewrite to kpre form. *)
  forevery_ext #(natlt rows)
    (fun (r : natlt rows) ->
       (x |-> Frac (1.0R /. SZ.v rows) (reveal sx)) **
       Cell output (abs_bij.gg r) |-> acc sout (abs_bij.gg r))
    (kpre_batched pre_map rows cols x output sx sout);
  ()
}

(* ── Ghost teardown ─────────────────────────────────────────────────────── *)

ghost
fn teardown_batched
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 rows cols)
  (#lout : layout1 rows)
  (x      : array2 et lin)
  (output : array1 et lout)
  (#sx   : EM.chest2 et rows cols)
  (#sout : chest1 et rows)
  ()
  norewrite
  requires
    (forall+ (r : natlt rows). kpost_batched pre_map rows cols x output sx sout r) **
    pure (SZ.fits (tlayout_ulen lout))
  ensures
    x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows pre_map sx)
{
  (* Unfold kpost to zip form. *)
  forevery_ext #(natlt rows)
    (kpost_batched pre_map rows cols x output sx sout)
    (fun (r : natlt rows) ->
       x |-> Frac (1.0R /. SZ.v rows) sx **
       Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> row_reduce pre_map sx r);

  (* Unzip. *)
  forevery_unzip #(natlt rows)
    (fun (_ : natlt rows) -> x |-> Frac (1.0R /. SZ.v rows) sx)
    (fun (r : natlt rows) -> Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> row_reduce pre_map sx r);

  (* Gather x. *)
  tensor_gather_n x rows;

  (* Rewrite output cells to use sout'. *)
  let sout' : chest1 et rows = hide (seq_to_chest1 (seq_reduce_rows pre_map sx));
  forevery_ext #(natlt rows)
    (fun (r : natlt rows) -> Cell output ((r, ()) <: abs (SZ.v rows @| INil)) |-> row_reduce pre_map sx r)
    (fun (r : natlt rows) -> Cell output (abs_bij.gg r) |-> acc (reveal sout') (abs_bij.gg r));

  forevery_iso_back (abs_bij #rows)
    (fun (i : abs (SZ.v rows @| INil)) -> Cell output i |-> acc (reveal sout') i);

  (* Implode output. *)
  tensor_implode output #1.0R #(reveal sout');
  ()
}

(* ── Kernel descriptor ──────────────────────────────────────────────────── *)

#push-options "--z3rlimit 40"
inline_for_extraction noextract
let kdesc_batched
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)              {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (#sx   : EM.chest2 et rows cols)
  (#sout : chest1 et rows)
  : kernel_desc
      (x |-> sx ** output |-> sout)
      (x |-> sx ** output |-> seq_to_chest1 (seq_reduce_rows pre_map sx)) =
{
  nthr     = rows;
  frame    = pure (SZ.fits (tlayout_ulen lout));
  setup    = setup_batched pre_map rows cols x output;
  teardown = teardown_batched pre_map rows cols x output;
  kpre     = kpre_batched  pre_map rows cols x output sx sout;
  kpost    = kpost_batched pre_map rows cols x output sx sout;
  f        = kf_batched    pre_map rows cols x output;
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_n _ _
#pop-options

(* ── Entry point ────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched
  (#et : Type0) {| scalar et |}
  (pre_map : et -> et)
  (rows : szp { SZ.v rows <= max_blocks * max_threads })
  (cols : szp)
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)              {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (#sx   : EM.chest2 et rows cols)
  (#sout : chest1 et rows)
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout)
  ensures
    on gpu_loc (output |-> seq_to_chest1 (seq_reduce_rows pre_map sx))
{
  launch_sync (kdesc_batched pre_map rows cols x output #sx #sout);
}
