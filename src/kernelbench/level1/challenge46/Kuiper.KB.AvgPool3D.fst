module Kuiper.KB.AvgPool3D

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout { is_full }
open Kuiper.Tensor.Layout.Alg { l2_row_major, l1_forward }
open Kuiper.Monoid.Reduce.F32 { reducer_fadd_f32 }
open Kuiper.Kernel.WindowReduce1D { windowreduce, windowreduce_result }
open Kuiper.Spec.Pool1D { pool_out_len_1d }
open Kuiper.Tensor.Layout.BCMPages { l2_bcm_pages, c_l2_bcm_pages }
open Kuiper.Array2.Recast { recast_gpu }
open Kuiper.Seq.Common { lseq_map }
module SZ = Kuiper.SizeT
module EM = Kuiper.EMatrix
module ML = FStar.Math.Lemmas
module SM = Kuiper.KB.ScalarMul

(* Verified 1-D pool output length used inside the complete entry, provably
   equal to the pure [pool_out_len_1d] specification. *)
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
fn avgpool3d_axis_fw
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
let avgpool3d_axis_fw_f32 =
  fun k s p d bc l l_out #_ #_ #_ #_ input output #fIn #sx #sout ->
    avgpool3d_axis_fw #f32 reducer_fadd_f32 k s p d bc l l_out input output
      #fIn #sx #sout

let avgpool3d_axis_fw_rm_f32 =
  fun k s p d bc l l_out input output #fIn #sx #sout ->
    avgpool3d_axis_fw_f32 k s p d bc l l_out
      #(l2_row_major bc l)     #_
      #(l2_row_major bc l_out) #_
      input output
      #fIn #sx #sout

(* ── Self-allocating per-axis entry (mirrors #44 [avgpool1d_alloc]) ───── *)

(* Reshape glue: a [(m, cn)] row-major Array2 buffer viewed as a flat
   [m*cn] Array1 over the same store, and back. *)
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
   matrices are equal. *)
let scale_matrix_cong (#m #cn:nat) (c1 c2 : f32) (e1 e2 : chest2 f32 m cn)
  : Lemma (requires c1 == c2 /\ e1 == e2)
          (ensures mk2 (fun (i:natlt m) (j:natlt cn) -> mul c1 (acc2 e1 i j))
                == mk2 (fun (i:natlt m) (j:natlt cn) -> mul c2 (acc2 e2 i j)))
  = ()

inline_for_extraction noextract
fn avgpool3d_axis_alloc
  (k s : szp)
  (p : sz)
  (d : szp)
  (bc : szp { SZ.v bc <= max_blocks * max_threads })
  (l : szp { SZ.fits (SZ.v bc * SZ.v l) })
  (input : array2 f32 (l2_row_major bc l) { is_global input })
  (#fIn : perm)
  (#sx  : chest2 f32 bc l)
  preserves
    cpu **
    on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (SZ.fits (SZ.v d * (SZ.v k - 1) + 1)) **
    pure (SZ.fits (SZ.v l + 2 * SZ.v p)) **
    pure (SZ.v d * (SZ.v k - 1) + 1 <= SZ.v l + 2 * SZ.v p) **
    pure (SZ.fits (pool_out_len_1d l k s p d
                     * SZ.v s + SZ.v k * SZ.v d)) **
    pure (SZ.fits (SZ.v bc *
            pool_out_len_1d l k s p d)) **
    pure (SZ.v bc *
            pool_out_len_1d l k s p d
          <= max_blocks * max_threads)
  returns r : (lo:sz { SZ.v lo == pool_out_len_1d l k s p d }
               & array2 f32 (l2_row_major bc lo))
  ensures
    on gpu_loc ((dsnd r) |->
      mk2 (fun (i:natlt bc) (j:natlt (dfst r)) ->
        mul (avgpool_recip_f32 k)
            (acc2 (windowreduce_result reducer_fadd_f32 sx
                       k s p d (dfst r)) i j))) **
    pure (SZ.v (dfst r) ==
            pool_out_len_1d l k s p d)
{
  let l_out = pool_out_len_1d_sz l k s p d;
  let output = alloc0 #f32 (bc *^ l_out) (l2_row_major bc l_out);
  avgpool3d_axis_fw_rm_f32 k s p d bc l l_out input output;
  (* output |-> wr, where wr is the per-window SUM. *)
  let inv_k = avgpool_recip_f32 k;
  let n : szp = bc *^ l_out;
  assert pure (SZ.v n == SZ.v bc * SZ.v l_out);
  let pp : erased nat = SZ.v n;
  let wr : chest2 f32 bc l_out =
    hide (windowreduce_result reducer_fadd_f32 sx
            k s p d l_out);
  (* View the row-major output buffer as a flat array1 over the same store. *)
  map_loc gpu_loc (fun () -> reshape2to1 pp output);
  (* Verified in-place /K scale on the flat view. *)
  SM.smul_fw_f32 inv_k n (from_array (l1_forward pp) (core output));
  (* Reflect the flat scale back to the matrix view. *)
  smul_reshape_eq2 inv_k #bc #l_out pp () (reveal wr);
  map_loc gpu_loc (fun () ->
    reshape1to2_eq pp output
      #(mk2 (fun (i:natlt bc) (j:natlt l_out) ->
          mul inv_k (acc2 (reveal wr) i j)))
      #_
      #(chest_map (mul inv_k)
          (from_seq (l1_forward pp)
             (to_seq (l2_row_major bc l_out) (reveal wr)))));
  scale_matrix_cong #bc #l_out
    inv_k (avgpool_recip_f32 k)
    (reveal wr)
    (windowreduce_result reducer_fadd_f32 sx
       k s p d l_out);
  rewrite
    (on gpu_loc (output |->
       mk2 (fun (i:natlt bc) (j:natlt l_out) ->
         mul inv_k (acc2 (reveal wr) i j))))
  as
    (on gpu_loc (output |->
       mk2 (fun (i:natlt bc) (j:natlt l_out) ->
         mul (avgpool_recip_f32 k)
             (acc2 (windowreduce_result reducer_fadd_f32 sx
                        k s p d l_out) i j))));
  (| (l_out <: (lo:sz { SZ.v lo == pool_out_len_1d l k s p d })), output |)
}

let avgpool3d_axis_alloc_f32 =
  fun k s p d bc l input #fIn #sx ->
    avgpool3d_axis_alloc k s p d bc l input #fIn #sx

let pool_out_len_1d_pos (l k s p d : nat)
  : Lemma (requires k >= 1 /\ s >= 1 /\ d >= 1 /\ d * (k - 1) + 1 <= l + 2 * p)
          (ensures pool_out_len_1d l k s p d >= 1)
  = ()

let prod3_comm (a x y : nat) : Lemma (a * x * y == a * y * x) = ()
let prod4_rotate (a b c d : nat)
  : Lemma (a * b * c * d == a * (d * c) * b) = ()

(* Layout-polymorphic version of the row-major reshape/scale proof above.
   Scaling every physical slot commutes with any full layout's bijection. *)
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
  (a:array2 f32 l { is_global a })
  (#s2:chest2 f32 m cn)
  preserves cpu
  requires on gpu_loc (a |-> s2) **
    pure (SZ.fits (SZ.v m * SZ.v cn)) **
    pure (SZ.v m * SZ.v cn <= max_blocks * max_threads) **
    pure (is_full_array (core a))
  ensures on gpu_loc (a |-> avgpool3d_scale_layout_result l c s2)
{
  let n : szp = m *^ cn;
  let pp : erased nat = SZ.v n;
  map_loc gpu_loc (fun () -> reshape_full2to1 pp a);
  SM.smul_fw_f32 c n (from_array (l1_forward pp) (core a));
  map_loc gpu_loc (fun () -> reshape1to_full2_physical pp a);
}

(* Keep the large nested composition symbols out of the main Pulse VC.  Each
   ghost bridge exposes only one opaque definition at one concrete boundary. *)
ghost
fn name_mid_w
  (bc depth h w:nat) (kw:szp) (sw:pos) (pw:nat) (dw:pos) (wo:pos)
  (a:array2 f32 (l2_bcm_pages (bc*depth) wo h))
  (#sx:chest2 f32 (bc*depth*h) w) (#f:perm)
  requires a |-> Frac f
    (from_seq (l2_bcm_pages (bc*depth) wo h)
      (to_seq (l2_row_major (bc*depth*h) wo)
        (avgpool3d_scale_layout_result (l2_row_major (bc*depth*h) wo)
          (avgpool_recip_f32 kw)
          (windowreduce_result reducer_fadd_f32 sx kw sw pw dw wo))))
  ensures a |-> Frac f (avgpool3d_mid_w_view bc depth h w kw sw pw dw wo sx)
{
  reveal_opaque (`%avgpool3d_axis_layout_result)
    (avgpool3d_axis_layout_result (l2_row_major (bc*depth*h) wo)
      kw sx sw pw dw);
  reveal_opaque (`%avgpool3d_mid_w_view)
    (avgpool3d_mid_w_view bc depth h w kw sw pw dw wo sx);
  rewrite
    (a |-> Frac f (from_seq (l2_bcm_pages (bc*depth) wo h)
      (to_seq (l2_row_major (bc*depth*h) wo)
        (avgpool3d_scale_layout_result (l2_row_major (bc*depth*h) wo)
          (avgpool_recip_f32 kw)
          (windowreduce_result reducer_fadd_f32 sx kw sw pw dw wo)))))
  as (a |-> Frac f (avgpool3d_mid_w_view bc depth h w kw sw pw dw wo sx));
}

ghost
fn name_mid_h
  (bc depth h w:nat) (kh kw:szp) (sh sw:pos) (ph pw:nat)
  (dh dw:pos) (wo ho:pos)
  (a:array2 f32 (l2_bcm_pages bc (ho*wo) depth))
  (#sx:chest2 f32 (bc*depth*h) w) (#f:perm)
  requires a |-> Frac f
    (from_seq (l2_bcm_pages bc (ho*wo) depth)
      (to_seq (l2_bcm_pages (bc*depth) wo ho)
        (avgpool3d_scale_layout_result (l2_bcm_pages (bc*depth) wo ho)
          (avgpool_recip_f32 kh)
          (windowreduce_result reducer_fadd_f32
            (avgpool3d_mid_w_view bc depth h w kw sw pw dw wo sx)
            kh sh ph dh ho))))
  ensures a |-> Frac f
    (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw wo ho sx)
{
  reveal_opaque (`%avgpool3d_axis_layout_result)
    (avgpool3d_axis_layout_result (l2_bcm_pages (bc*depth) wo ho) kh
      (avgpool3d_mid_w_view bc depth h w kw sw pw dw wo sx) sh ph dh);
  reveal_opaque (`%avgpool3d_mid_h_view)
    (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw wo ho sx);
  rewrite
    (a |-> Frac f (from_seq (l2_bcm_pages bc (ho*wo) depth)
      (to_seq (l2_bcm_pages (bc*depth) wo ho)
        (avgpool3d_scale_layout_result (l2_bcm_pages (bc*depth) wo ho)
          (avgpool_recip_f32 kh)
          (windowreduce_result reducer_fadd_f32
            (avgpool3d_mid_w_view bc depth h w kw sw pw dw wo sx)
            kh sh ph dh ho)))))
  as (a |-> Frac f
    (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw wo ho sx));
}

ghost
fn name_final
  (bc depth h w:nat) (kd kh kw:szp) (sd sh sw:pos) (pd ph pw:nat)
  (dd dh dw:pos) (wo ho do_:pos)
  (a:array2 f32 (l2_bcm_pages bc (ho*wo) do_))
  (#sx:chest2 f32 (bc*depth*h) w) (#f:perm)
  requires a |-> Frac f
    (avgpool3d_scale_layout_result (l2_bcm_pages bc (ho*wo) do_)
      (avgpool_recip_f32 kd)
      (windowreduce_result reducer_fadd_f32
        (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw wo ho sx)
        kd sd pd dd do_))
  ensures a |-> Frac f
    (avgpool3d_axis_layout_result (l2_bcm_pages bc (ho*wo) do_) kd
      (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw wo ho sx)
      sd pd dd)
{
  reveal_opaque (`%avgpool3d_axis_layout_result)
    (avgpool3d_axis_layout_result (l2_bcm_pages bc (ho*wo) do_) kd
      (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw wo ho sx)
      sd pd dd);
  rewrite
    (a |-> Frac f (avgpool3d_scale_layout_result (l2_bcm_pages bc (ho*wo) do_)
      (avgpool_recip_f32 kd)
      (windowreduce_result reducer_fadd_f32
        (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw wo ho sx)
        kd sd pd dd do_)))
  as (a |-> Frac f
    (avgpool3d_axis_layout_result (l2_bcm_pages bc (ho*wo) do_) kd
      (avgpool3d_mid_h_view bc depth h w kh kw sh sw ph pw dh dw wo ho sx)
      sd pd dd));
}

#push-options "--z3rlimit 60"
fn avgpool3d_full_alloc_f32
  (kd kh kw sd sh sw : szp) (pd ph pw : sz) (dd dh dw : szp)
  (bc depth h w : szp)
  (#sq_bd : squash (SZ.fits (SZ.v bc * SZ.v depth)))
  (#sq_bdh : squash (SZ.fits (SZ.v bc * SZ.v depth * SZ.v h)))
  (input : array2 f32 (l2_row_major (bc * depth * h) w) { is_global input })
  (#fIn : perm) (#sx : chest2 f32 (bc * depth * h) w)
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (avgpool3d_full_pre kd kh kw sd sh sw pd ph pw dd dh dw
      bc depth h w)
  returns r : avgpool3d_full_result kd kh kw sd sh sw pd ph pw dd dh dw
    bc depth h w
  ensures avgpool3d_full_post kd kh kw sd sh sw pd ph pw dd dh dw
    bc depth h w sx r
{
  let wo0 = pool_out_len_1d_sz w kw sw pw dw;
  pool_out_len_1d_pos w kw sw pw dw;
  let wo : (x:sz { SZ.v x == pool_out_len_1d w kw sw pw dw /\ SZ.v x > 0 }) = wo0;
  let rows_w : szp = (bc *^ depth) *^ h;
  let mid_w = alloc0 #f32 (rows_w *^ wo) (l2_row_major rows_w wo);
  avgpool3d_axis_fw_rm_f32 kw sw pw dw rows_w w wo input mid_w;
  scale_full2_f32 (avgpool_recip_f32 kw) rows_w wo mid_w;

  prod3_comm (SZ.v bc * SZ.v depth) (SZ.v h) (SZ.v wo);
  let mid_h_in = recast_gpu
    (l2_bcm_pages (SZ.v bc * SZ.v depth) (SZ.v wo) (SZ.v h)) mid_w;
  map_loc gpu_loc (fun () -> name_mid_w (SZ.v bc) (SZ.v depth) (SZ.v h)
    (SZ.v w) kw (SZ.v sw) (SZ.v pw) (SZ.v dw) (SZ.v wo) mid_h_in #sx #1.0R);
  let ho0 = pool_out_len_1d_sz h kh sh ph dh;
  pool_out_len_1d_pos h kh sh ph dh;
  let ho : (x:sz { SZ.v x == pool_out_len_1d h kh sh ph dh /\ SZ.v x > 0 }) = ho0;
  let rows_h : szp = (bc *^ depth) *^ wo;
  let mid_h = alloc0 #f32 (rows_h *^ ho)
    (l2_bcm_pages (SZ.v bc * SZ.v depth) (SZ.v wo) (SZ.v ho));
  avgpool3d_axis_fw #f32 reducer_fadd_f32 kh sh ph dh rows_h h ho
    #(l2_bcm_pages (SZ.v bc * SZ.v depth) (SZ.v wo) (SZ.v h))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc * SZ.v depth)) wo h)
    #(l2_bcm_pages (SZ.v bc * SZ.v depth) (SZ.v wo) (SZ.v ho))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc * SZ.v depth)) wo ho)
    mid_h_in mid_h;
  free mid_h_in;
  scale_full2_f32 (avgpool_recip_f32 kh) rows_h ho mid_h;

  prod4_rotate (SZ.v bc) (SZ.v depth) (SZ.v wo) (SZ.v ho);
  let mid_d_in = recast_gpu
    (l2_bcm_pages (SZ.v bc) (SZ.v ho * SZ.v wo) (SZ.v depth)) mid_h;
  map_loc gpu_loc (fun () -> name_mid_h (SZ.v bc) (SZ.v depth) (SZ.v h)
    (SZ.v w) kh kw (SZ.v sh) (SZ.v sw) (SZ.v ph) (SZ.v pw)
    (SZ.v dh) (SZ.v dw) (SZ.v wo) (SZ.v ho) mid_d_in #sx #1.0R);
  let do0 = pool_out_len_1d_sz depth kd sd pd dd;
  pool_out_len_1d_pos depth kd sd pd dd;
  let do_ : (x:sz { SZ.v x == pool_out_len_1d depth kd sd pd dd /\ SZ.v x > 0 }) = do0;
  let rows_d : szp = bc *^ (ho *^ wo);
  let out = alloc0 #f32 (rows_d *^ do_)
    (l2_bcm_pages (SZ.v bc) (SZ.v ho * SZ.v wo) (SZ.v do_));
  avgpool3d_axis_fw #f32 reducer_fadd_f32 kd sd pd dd rows_d depth do_
    #(l2_bcm_pages (SZ.v bc) (SZ.v ho * SZ.v wo) (SZ.v depth))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc)) (ho *^ wo) depth)
    #(l2_bcm_pages (SZ.v bc) (SZ.v ho * SZ.v wo) (SZ.v do_))
    #(c_l2_bcm_pages (FStar.Ghost.hide (SZ.v bc)) (ho *^ wo) do_)
    mid_d_in out;
  free mid_d_in;
  scale_full2_f32 (avgpool_recip_f32 kd) rows_d do_ out;
  map_loc gpu_loc (fun () -> name_final (SZ.v bc) (SZ.v depth) (SZ.v h)
    (SZ.v w) kd kh kw (SZ.v sd) (SZ.v sh) (SZ.v sw) (SZ.v pd)
    (SZ.v ph) (SZ.v pw) (SZ.v dd) (SZ.v dh) (SZ.v dw)
    (SZ.v wo) (SZ.v ho) (SZ.v do_) out #sx #1.0R);
  (| wo, (| ho, (| do_, out |) |) |)
}
#pop-options

#push-options "--z3rlimit 60"
fn avgpool3d_raw_alloc_f32
  (k s : szp) (p : sz) (b : szp)
  (c : szp { SZ.fits (SZ.v b * SZ.v c) })
  (depth h w : szp)
  (#sq_bd : squash (SZ.fits (SZ.v b * SZ.v c * SZ.v depth)))
  (#sq_bdh : squash (SZ.fits (SZ.v b * SZ.v c * SZ.v depth * SZ.v h)))
  (input : array2 f32 (l2_row_major (b * c * depth * h) w) { is_global input })
  (#fIn : perm)
  (#sx : chest2 f32 (b * c * depth * h) w)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fIn sx)
  requires
    pure (avgpool3d_full_pre k k k s s s p p p 1 1 1
      (SZ.v (b *^ c)) depth h w)
  returns r : avgpool3d_full_result k k k s s s p p p 1 1 1
    (SZ.v (b *^ c)) depth h w
  ensures avgpool3d_full_post k k k s s s p p p 1 1 1
    (SZ.v (b *^ c)) depth h w sx r
{
  avgpool3d_full_alloc_f32 k k k s s s p p p 1sz 1sz 1sz
    (b *^ c) depth h w #sq_bd #sq_bdh input #fIn #sx
}
#pop-options
