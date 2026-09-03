module Kuiper.KB.AvgPool2D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major, l1_forward }
open Kuiper.Monoid.Reduce.F32 { reducer_fadd_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce, windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
open Kuiper.Seq.Common { lseq_map }
open Kuiper.Shareable
open Kuiper.Tensor.Layout.BCMPages { l2_bcm_pages, c_l2_bcm_pages }
open Kuiper.Array2.Recast { recast_gpu }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module ML = FStar.Math.Lemmas
module SM = Kuiper.KB.ScalarMul
module TMap = Kuiper.Kernel.TMap

(* Verified 1-D pool output length used by the internal axis building block,
   provably equal to the pure [pool_out_len_1d] specification. *)
let pool_out_len_1d_sz
  (l k s : szp) (p : sz) (d : szp)
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

(* Verified, extractable reciprocal 1/k as f32 (extracts to
   1.0f / (float)(int64_t)(uint64_t)k); the per-axis average divisor is
   computed inside the verification boundary. *)
let avgpool_recip_f32 (k : szp)
  : r:f32 { r %~ (1.0R /. FStar.Real.of_int (SZ.v k)) }
  = let k_i64 = FStar.Int.Cast.uint64_to_int64
      (FStar.SizeT.sizet_to_uint64 k) in
    assert (FStar.Int64.v k_i64 == SZ.v k);
    of_int_approx #f32 k_i64;
    assert ((one #f32) %~ 1.0R);
    div_approx (one #f32) (of_int #f32 k_i64)
      1.0R (FStar.Real.of_int (SZ.v k));
    let r : f32 = div (one #f32) (of_int #f32 k_i64) in
    assert (r %~ (1.0R /. FStar.Real.of_int (SZ.v k)));
    r

inline_for_extraction noextract
fn avgpool2d_axis_fw
  (#t : Type0) {| scalar t |}
  (m_inst : Kuiper.Monoid.Reduce.reducer t)
  (k s : szp)
  (p : sz)
  (d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l    : szp)
  (l_out : sz { SZ.v l_out == pool_out_len_1d l k s p d })
  (#lin  : layout2 bc l)     {| ctlayout lin  |}
  (#lout : layout2 bc l_out) {| ctlayout lout |}
  (input  : array2 t lin  { is_global input  })
  (output : array2 t lout { is_global output })
  (#fIn  : perm)
  (#sx   : chest2 t bc l)
  (#sout : chest2 t bc l_out)
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
let avgpool2d_axis_fw_f32 =
  fun k s p d bc l l_out #_ #_ #_ #_ input output #fIn #sx #sout ->
    avgpool2d_axis_fw #f32 reducer_fadd_f32 k s p d bc l l_out input output
      #fIn #sx #sout

let avgpool2d_axis_fw_rm_f32 =
  fun k s p d bc l l_out input output #fIn #sx #sout ->
    avgpool2d_axis_fw_f32 k s p d bc l l_out
      #(l2_row_major bc l)     #_
      #(l2_row_major bc l_out) #_
      input output
      #fIn #sx #sout

(* ── Self-allocating per-axis entry (mirrors #44 [avgpool1d_alloc]) ───── *)

(* Reshape glue: a [(m, cn)] row-major Array2 buffer viewed as a flat
   [m*cn] Array1 over the same store, and back.  [l1_forward]/[l2_row_major]
   [from_seq]/[to_seq] are inverse via the library round-trip lemmas. *)
ghost
fn reshape2to1
  (#et:Type) (#m #cn:nat)
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 et (l2_row_major m cn))
  (#s2 : chest2 et m cn)
  (#f : perm)
  requires
    a2 |-> Frac f s2
  ensures
    from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l2_row_major m cn) s2))
{
  tensor_concr a2;
  tensor_abs' (l1_forward p) (core a2)
}

ghost
fn reshape1to2
  (#et:Type) (#m #cn:nat)
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 et (l2_row_major m cn))
  (#s2 : chest2 et m cn)
  (#f : perm)
  requires
    from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l2_row_major m cn) s2))
  ensures
    a2 |-> Frac f s2
{
  tensor_concr (from_array (l1_forward p) (core a2));
  rewrite
    (core (from_array (l1_forward p) (core a2))
      |-> Frac f (to_seq (l1_forward p)
                    (from_seq (l1_forward p)
                       (to_seq (l2_row_major m cn) s2))))
  as
    (core a2 |-> Frac f (to_seq (l2_row_major m cn) s2));
  tensor_abs (l2_row_major m cn) (core a2) #f #s2;
  rewrite
    (from_array (l2_row_major m cn) (core a2) |-> Frac f s2)
  as
    (a2 |-> Frac f s2);
}

ghost
fn reshape1to2_eq
  (#et:Type) (#m #cn:nat)
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 et (l2_row_major m cn))
  (#s2 : chest2 et m cn)
  (#f : perm)
  (#e : chest1 et p)
  (#_ : squash (
     e == from_seq (l1_forward p)
            (to_seq (l2_row_major m cn) s2)))
  requires
    from_array (l1_forward p) (core a2) |-> Frac f e
  ensures
    a2 |-> Frac f s2
{
  rewrite
    (from_array (l1_forward p) (core a2) |-> Frac f e)
  as
    (from_array (l1_forward p) (core a2)
      |-> Frac f (from_seq (l1_forward p)
                     (to_seq (l2_row_major m cn) s2)));
  reshape1to2 p a2 #s2 #f;
}

(* Mapping [mul c] over the flattened row-major sequence equals flattening the
   scaled matrix (pure index reasoning; both sides are length-[p] sequences). *)
#push-options ""
let smul_reshape_eq2
  (c : f32) (#m #cn : nat) (p:nat) (_:squash (p == m * cn))
  (s2 : chest2 f32 m cn)
  : Lemma
    (chest_map (mul c)
        (from_seq (l1_forward p) (to_seq (l2_row_major m cn) s2))
     == from_seq (l1_forward p)
           (to_seq (l2_row_major m cn)
              (mk2 (fun (i:natlt m) (j:natlt cn) -> mul c (acc2 s2 i j)))))
  = let l2 = l2_row_major m cn in
    let l1 = l1_forward p in
    let scaled = mk2 (fun (i:natlt m) (j:natlt cn) -> mul c (acc2 s2 i j)) in
    let lhs = chest_map (mul c) (from_seq l1 (to_seq l2 s2)) in
    let rhs = from_seq l1 (to_seq l2 scaled) in
    let aux (i:abs (p @| INil)) : Lemma (acc lhs i == acc rhs i) =
      let q = l1.imap.f i in
      ()
    in
    Classical.forall_intro aux;
    Kuiper.Chest.lemma_equal_intro lhs rhs;
    Kuiper.Chest.ext lhs rhs
#pop-options

(* Congruence: scaling-matrices built from equal scalars and equal source
   matrices are equal (close the post by reflexivity once arguments equal). *)
let scale_matrix_cong (#m #cn:nat) (c1 c2 : f32) (e1 e2 : chest2 f32 m cn)
  : Lemma (requires c1 == c2 /\ e1 == e2)
          (ensures mk2 (fun (i:natlt m) (j:natlt cn) -> mul c1 (acc2 e1 i j))
                == mk2 (fun (i:natlt m) (j:natlt cn) -> mul c2 (acc2 e2 i j)))
  = ()

let prod3_comm (a x y : nat) : Lemma (a * x * y == a * y * x) = ()

(* A full Array2 may be scaled through its physical flat sequence.  This is
   exactly the byte order later preserved by the zero-copy layout recast. *)
ghost
fn reshape_full2to1
  (#m #cn:nat) (#l:layout2 m cn { is_full l })
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 f32 l) (#s2 : chest2 f32 m cn) (#f : perm)
  requires a2 |-> Frac f s2
  ensures from_array (l1_forward p) (core a2)
    |-> Frac f (from_seq (l1_forward p) (to_seq l s2))
{
  tensor_concr a2;
  tensor_abs' (l1_forward p) (core a2)
}

ghost
fn reshape1to_full2_physical
  (#m #cn:nat) (#l:layout2 m cn { is_full l })
  (p:nat) (#_ : squash (p == m * cn))
  (a2 : array2 f32 l) (#e : chest1 f32 p) (#f : perm)
  requires from_array (l1_forward p) (core a2) |-> Frac f e
  ensures a2 |-> Frac f (from_seq l (to_seq (l1_forward p) e))
{
  tensor_concr (from_array (l1_forward p) (core a2));
  rewrite (core (from_array (l1_forward p) (core a2))
      |-> Frac f (to_seq (l1_forward p) e))
    as (core a2 |-> Frac f (to_seq (l1_forward p) e));
  tensor_abs' l (core a2);
  rewrite (from_array l (core a2) |->
      Frac f (from_seq l (to_seq (l1_forward p) e)))
    as (a2 |-> Frac f (from_seq l (to_seq (l1_forward p) e)));
}

inline_for_extraction noextract
fn scale_full2_f32
  (c:f32) (m cn:szp) (#l:layout2 m cn { is_full l })
  (a:array2 f32 l { is_global a }) (#s2:chest2 f32 m cn)
  preserves cpu
  requires on gpu_loc (a |-> s2) **
    pure (SZ.fits (SZ.v m * SZ.v cn)) **
    pure (SZ.v m * SZ.v cn <= max_blocks * max_threads) **
    pure (is_full_array (core a))
  ensures on gpu_loc (a |-> avgpool2d_scale_layout_result l c s2)
{
  let n : szp = m *^ cn;
  let pp : erased nat = SZ.v n;
  map_loc gpu_loc (fun () -> reshape_full2to1 pp a);
  SM.smul_fw_f32 c n (from_array (l1_forward pp) (core a));
  map_loc gpu_loc (fun () -> reshape1to_full2_physical pp a);
}

(* Give the two physical-layout boundaries compact logical names.  Keeping
   these rewrites in small ghost functions prevents the complete entry's VC
   from repeatedly expanding the nested pooling result. *)
ghost
fn name_avgpool2d_mid_w
  (bc h w:nat)
  (a:array2 f32 (l2_bcm_pages bc 186 h))
  (#sx:chest2 f32 (bc * h) w) (#f:perm)
  requires a |-> Frac f
    (from_seq (l2_bcm_pages bc 186 h)
      (to_seq (l2_row_major (bc * h) 186)
        (avgpool2d_scale_layout_result (l2_row_major (bc * h) 186)
          (avgpool_recip_f32 11sz)
          (windowreduce_result reducer_fadd_f32 sx 11 11 0 1 186))))
  ensures a |-> Frac f (avgpool2d_mid_w_view bc h w sx)
{
  reveal_opaque (`%avgpool2d_axis_layout_result)
    (avgpool2d_axis_layout_result (l2_row_major (bc * h) 186)
      11sz sx 11 0 1);
  reveal_opaque (`%avgpool2d_mid_w_view)
    (avgpool2d_mid_w_view bc h w sx);
  rewrite
    (a |-> Frac f
      (from_seq (l2_bcm_pages bc 186 h)
        (to_seq (l2_row_major (bc * h) 186)
          (avgpool2d_scale_layout_result (l2_row_major (bc * h) 186)
            (avgpool_recip_f32 11sz)
            (windowreduce_result reducer_fadd_f32 sx 11 11 0 1 186)))))
  as (a |-> Frac f (avgpool2d_mid_w_view bc h w sx));
}

ghost
fn name_avgpool2d_half_result
  (bc h w:nat)
  (a:array2 f32 (l2_bcm_pages bc 186 186))
  (#sx:chest2 f32 (bc * h) w) (#f:perm)
  requires a |-> Frac f
    (avgpool2d_scale_layout_result (l2_bcm_pages bc 186 186)
      (avgpool_recip_f32 11sz)
      (windowreduce_result reducer_fadd_f32
        (avgpool2d_mid_w_view bc h w sx)
        11 11 0 1 186))
  ensures a |-> Frac f (avgpool2d_half_result bc h w sx)
{
  reveal_opaque (`%avgpool2d_axis_layout_result)
    (avgpool2d_axis_layout_result (l2_bcm_pages bc 186 186)
      11sz (avgpool2d_mid_w_view bc h w sx) 11 0 1);
  reveal_opaque (`%avgpool2d_half_result)
    (avgpool2d_half_result bc h w sx);
  rewrite
    (a |-> Frac f
      (avgpool2d_scale_layout_result (l2_bcm_pages bc 186 186)
        (avgpool_recip_f32 11sz)
        (windowreduce_result reducer_fadd_f32
          (avgpool2d_mid_w_view bc h w sx)
          11 11 0 1 186)))
  as (a |-> Frac f (avgpool2d_half_result bc h w sx));
}

(* One complete 2-D average-pool over an ABI-sized batch/channel half.  Keep
   this as an extracted helper so its launch-local stream is scoped inside the
   helper rather than duplicated into the full entry by extraction. *)
fn avgpool2d_half_alloc
  (bc : szp { SZ.v bc == 512 })
  (h : szp { SZ.v h == 2048 })
  (w : szp { SZ.v w == 2048 })
  (input : array2 f32 (l2_row_major (bc * h) w) { is_global input })
  (#fIn:perm) (#sx:chest2 f32 (bc * h) w)
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  returns out : array2 f32 (l2_bcm_pages (SZ.v bc) 186 186)
  ensures on gpu_loc (out |-> avgpool2d_half_result (SZ.v bc) (SZ.v h) (SZ.v w) sx) **
    pure (is_global out) ** pure (is_full_array (core out))
{
  let w_out : szp = 186sz;
  let rows_w : szp = bc *^ h;
  let mid_w = alloc0 #f32 (rows_w *^ w_out) (l2_row_major rows_w w_out);
  avgpool2d_axis_fw_rm_f32 11sz 11sz 0sz 1sz rows_w w w_out input mid_w;
  scale_full2_f32 (avgpool_recip_f32 11sz) rows_w w_out mid_w;

  prod3_comm (SZ.v bc) (SZ.v h) 186;
  let mid_h_in = recast_gpu (l2_bcm_pages (SZ.v bc) 186 (SZ.v h)) mid_w;
  map_loc gpu_loc (fun () ->
    name_avgpool2d_mid_w (SZ.v bc) (SZ.v h) (SZ.v w)
      mid_h_in #sx #1.0R);
  let h_out : szp = 186sz;
  let rows_h : szp = bc *^ w_out;
  let out = alloc0 #f32 (rows_h *^ h_out)
    (l2_bcm_pages (SZ.v bc) (SZ.v w_out) (SZ.v h_out));
  avgpool2d_axis_fw #f32 reducer_fadd_f32 11sz 11sz 0sz 1sz
    rows_h h h_out
    #(l2_bcm_pages (SZ.v bc) (SZ.v w_out) (SZ.v h))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc)) w_out h)
    #(l2_bcm_pages (SZ.v bc) (SZ.v w_out) (SZ.v h_out))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc)) w_out h_out)
    mid_h_in out;
  free mid_h_in;
  scale_full2_f32 (avgpool_recip_f32 11sz) rows_h h_out out;
  map_loc gpu_loc (fun () ->
    name_avgpool2d_half_result (SZ.v bc) (SZ.v h) (SZ.v w)
      out #sx #1.0R);
  out
}

inline_for_extraction noextract unfold
let concat_shape (n:nat) : shape 1 = (n + n) @| INil
inline_for_extraction noextract
let concat_index (#n:nat) (i:abs (concat_shape n)) : natlt (n+n) =
  let (j, ()) = i in j

inline_for_extraction noextract unfold
let cconcat_index (#n:erased nat) (i:conc (concat_shape n)) : szlt (n+n) =
  let (j, ()) = i in j

let cconcat_index_up (#n:erased nat) (i:conc (concat_shape n))
  : Lemma (cconcat_index i == SZ.uint_to_t (concat_index (up i)))
  = ()

inline_for_extraction noextract
fn concat_ff
  (#n:erased nat)
  (n_sz:szp { SZ.v n_sz == n })
  (a0 a1:array1 f32 (l1_forward n))
  (#s0 #s1:chest1 f32 n) (#f0 #f1 #fr:perm)
  (i:conc (concat_shape n)) (_:f32)
  norewrite
  preserves gpu **
    (a0 |-> Frac (f0 *. fr) s0) ** (a1 |-> Frac (f1 *. fr) s1)
  returns r:f32
  ensures pure (r ==
    (if concat_index (up i) < n
     then acc1 s0 (concat_index (up i))
     else acc1 s1 (concat_index (up i) - n)))
{
  let ii = cconcat_index i;
  if SZ.(ii <^ n_sz) {
    let ii0 = SZ.(ii %^ n_sz);
    tensor_read a0 (ii0, ())
  } else {
    let jj = SZ.((ii -^ n_sz) %^ n_sz);
    tensor_read a1 (jj, ())
  }
}

inline_for_extraction noextract
fn concat2_gpu
  (n:szp { SZ.v n + SZ.v n <= max_blocks * max_threads })
  (a0:array1 f32 (l1_forward n) { is_global a0 })
  (a1:array1 f32 (l1_forward n) { is_global a1 })
  (out:array1 f32 (l1_forward (n+n)) { is_global out })
  (#s0 #s1:chest1 f32 n) (#so:chest1 f32 (n+n)) (#f0 #f1:perm)
  norewrite
  preserves cpu ** on gpu_loc (a0 |-> Frac f0 s0) **
    on gpu_loc (a1 |-> Frac f1 s1)
  requires on gpu_loc (out |-> so)
  ensures on gpu_loc (out |-> avgpool2d_concat_result s0 s1)
{
  let ntotal : szp = n *^ 2sz;
  launch_sync (TMap.kmap
    (CCons ntotal CNil)
    (fun fr -> (a0 |-> Frac (f0 *. fr) s0) **
               (a1 |-> Frac (f1 *. fr) s1))
    #(double_shareable (fun fr -> a0 |-> Frac fr s0)
                       (fun fr -> a1 |-> Frac fr s1) f0 f1)
    (fun i _ r -> r ==
      (if concat_index i < n then acc1 s0 (concat_index i)
       else acc1 s1 (concat_index i - n)))
    (concat_ff n a0 a1)
    ntotal out #so #_ #1.0R);
  with so'. assert on gpu_loc (out |-> so');
  assert pure (equal so' (avgpool2d_concat_result s0 s1));
}

(* These are verification-only names for the canonical challenge dimensions.
   They inline at extraction, so they introduce no public ABI parameters. *)
inline_for_extraction noextract
let avgpool2d_canonical_bc : x:szp { SZ.v x == 512 } = 512sz

inline_for_extraction noextract
let avgpool2d_canonical_h : x:szp { SZ.v x == 2048 } = 2048sz

inline_for_extraction noextract
let avgpool2d_canonical_w : x:szp { SZ.v x == 2048 } = 2048sz

inline_for_extraction noextract
let avgpool2d_canonical_half_n
  : x:szp { SZ.v x == 512 * 186 * 186 }
  = (avgpool2d_canonical_bc *^ 186sz) *^ 186sz

#push-options "--z3rlimit 60"
fn avgpool2d_full_alloc_f32
  (input0 : array2 f32
    (l2_row_major (avgpool2d_canonical_bc * avgpool2d_canonical_h)
      avgpool2d_canonical_w) { is_global input0 })
  (input1 : array2 f32
    (l2_row_major (avgpool2d_canonical_bc * avgpool2d_canonical_h)
      avgpool2d_canonical_w) { is_global input1 })
  (#f0 #f1 : perm)
  (#sx0 #sx1 : chest2 f32
    (avgpool2d_canonical_bc * avgpool2d_canonical_h)
    avgpool2d_canonical_w)
  preserves cpu ** on gpu_loc (input0 |-> Frac f0 sx0) **
    on gpu_loc (input1 |-> Frac f1 sx1)
  returns out : array1 f32
    (l1_forward
      (SZ.v avgpool2d_canonical_half_n + SZ.v avgpool2d_canonical_half_n))
  ensures on gpu_loc (out |->
    avgpool2d_concat_result
      (from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
        (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
          (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
            (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx0)))
      (from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
        (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
          (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
            (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx1))))
{
  let out0 = avgpool2d_half_alloc avgpool2d_canonical_bc
    avgpool2d_canonical_h avgpool2d_canonical_w input0;
  let out1 = avgpool2d_half_alloc avgpool2d_canonical_bc
    avgpool2d_canonical_h avgpool2d_canonical_w input1;
  map_loc gpu_loc (fun () ->
    reshape_full2to1 (SZ.v avgpool2d_canonical_half_n) out0);
  map_loc gpu_loc (fun () ->
    reshape_full2to1 (SZ.v avgpool2d_canonical_half_n) out1);
  let flat0 = from_array
    (l1_forward (SZ.v avgpool2d_canonical_half_n)) (core out0);
  let flat1 = from_array
    (l1_forward (SZ.v avgpool2d_canonical_half_n)) (core out1);
  rewrite
    (on gpu_loc
      (from_array (l1_forward (SZ.v avgpool2d_canonical_half_n)) (core out0) |->
        from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
          (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
            (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
              (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx0))))
  as
    (on gpu_loc
      (flat0 |->
        from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
          (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
            (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
              (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx0))));
  rewrite
    (on gpu_loc
      (from_array (l1_forward (SZ.v avgpool2d_canonical_half_n)) (core out1) |->
        from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
          (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
            (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
              (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx1))))
  as
    (on gpu_loc
      (flat1 |->
        from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
          (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
            (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
              (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx1))));
  let out = alloc0 #f32
    (avgpool2d_canonical_half_n +^ avgpool2d_canonical_half_n)
    (l1_forward
      (SZ.v avgpool2d_canonical_half_n + SZ.v avgpool2d_canonical_half_n));
  concat2_gpu avgpool2d_canonical_half_n flat0 flat1 out
    #(from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
      (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
        (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
          (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx0)))
    #(from_seq (l1_forward (SZ.v avgpool2d_canonical_half_n))
      (to_seq (l2_bcm_pages (SZ.v avgpool2d_canonical_bc) 186 186)
        (avgpool2d_half_result (SZ.v avgpool2d_canonical_bc)
          (SZ.v avgpool2d_canonical_h) (SZ.v avgpool2d_canonical_w) sx1)))
    #_ #1.0R #1.0R;
  free flat0;
  free flat1;
  out
}
#pop-options
