module Kuiper.KB.BatchNorm

#lang-pulse
open Kuiper
open Kuiper.Scalars.Ops
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor.Layout.BCMChannels
open Kuiper.Approximates.Base
open Kuiper.EMatrix
open Kuiper.Spec.Frobenius
open Kuiper.Spec.BatchNorm
module SZ = Kuiper.SizeT
module Map = Kuiper.Kernel.Map
module HRed = Kuiper.Kernel.HReduce
module KS = Kuiper.Seq.Common
module Tensor = Kuiper.Tensor
open Kuiper.Scalars
open Pulse.Lib.Trade

(* ─────────────────────────────────────────────────────────────────────
   Bridging lemmas and a scalar-read helper for the migration off the
   deleted [Kuiper.Array1]/[Kuiper.Array2] modules onto [Kuiper.Tensor] +
   [chest].  These mirror the green siblings [Kuiper.KB.MeanVarNorm] and
   [Kuiper.Kernel.BatchedGEMM].
   ───────────────────────────────────────────────────────────────────── *)

(* [seq_map id] is the identity. *)
let seq_map_id_eq (#a:Type) (s : Seq.seq a)
  : Lemma (Seq.equal (KS.seq_map id s) s)
  = ()

(* [chest1_to_seq] commutes with [chest_map]/[seq_map]. *)
let chest_map_to_seq (#et1 #et2 : Type) (#nn : nat)
  (f : et1 -> et2) (c : chest1 et1 nn)
  : Lemma (chest1_to_seq (chest_map f c) == KS.seq_map f (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (chest_map f c))
                      (KS.seq_map f (chest1_to_seq c)))

(* [chest1_to_seq] commutes with [to_real_chest]/[to_real_seq]. *)
let to_real_chest_to_seq (#et : Type0) {| scalar et, real_like et |} (#nn : nat)
  (c : chest1 et nn)
  : Lemma (chest1_to_seq (to_real_chest c) == to_real_seq (chest1_to_seq c))
  = assert (Seq.equal (chest1_to_seq (to_real_chest c))
                      (to_real_seq (chest1_to_seq c)))

(* [to_seq] on [l1_forward] equals [chest1_to_seq]. *)
let lem_to_seq (#et:Type) (nn:nat) (c : chest1 et nn)
  : Lemma (to_seq (l1_forward nn) c == chest1_to_seq c)
  = assert (Seq.equal (to_seq (l1_forward nn) c) (chest1_to_seq c))

(* The generic concrete instance for [tlayout_slice] must inspect a
   dependently shaped index.  Karamel consequently lowers its rank-1 index to
   [any] when this slice is used inside an extracted map kernel.  This
   specialization keeps the index shape concrete and delegates directly to
   the BCM-channels cimap. *)
#push-options "--z3rlimit 100 --fuel 4 --ifuel 3"
inline_for_extraction noextract
instance c_bcm_channel_slice
  (n  : erased nat)
  (c  : szp)
  (hw : szp { SZ.v hw > 0 /\
              SZ.fits (n * SZ.v hw) /\
              SZ.fits (SZ.v hw * SZ.v c) /\
              SZ.fits (n * (SZ.v hw * SZ.v c)) })
  (ci : szlt c)
  : ctlayout #1 #((n * SZ.v hw) @| INil)
      (tlayout_slice (l2_bcm_channels n (SZ.v c) (SZ.v hw)) 0 (SZ.v ci))
  = {
      ulen_fits = ();
      all_fit = ();
      cimap = fun (idx : conc ((n * SZ.v hw) @| INil)) ->
        [@@inline_let] let (k, ()) = idx in
        (c_l2_bcm_channels n c hw).cimap (ci, (k, ()))
    }
#pop-options

#push-options "--fuel 6 --ifuel 4"
let acc1_chest1_to_seq (#et:Type) (#nn:nat) (c:chest1 et nn) (i:natlt nn)
  : Lemma (Seq.index (chest1_to_seq c) i == acc1 c i)
  = ()
#pop-options

(* A cell of a chest2 row equals the matrix cell (fuel to compute through
   the [chest_slice] bijection). *)
#push-options "--fuel 6 --ifuel 4"
let acc1_chest2_row (#et:Type) (#r #cc:nat) (x:chest2 et r cc)
  (i:natlt r) (j:natlt cc)
  : Lemma (acc1 (chest2_row x i) j == acc2 x i j)
  = ()
#pop-options

(* [chest1_to_seq] of a chest2 row equals [ematrix_row]. *)
let chest2_row_to_seq (#et:Type) (#r #cc:nat) (x:chest2 et r cc) (i:natlt r)
  : Lemma (chest1_to_seq (chest2_row x i) == ematrix_row x i)
  = Classical.forall_intro (acc1_chest2_row x i);
    Seq.lemma_eq_intro (chest1_to_seq (chest2_row x i)) (ematrix_row x i)

(* A cell of [chest_update_slice 0 ci x newr]. *)
#push-options "--fuel 6 --ifuel 4"
let acc2_chest_update_slice0 (#et:Type) (#r #cc:nat)
  (x:chest2 et r cc) (ci:natlt r) (newr:chest1 et cc) (i:natlt r) (j:natlt cc)
  : Lemma (acc2 (chest_update_slice 0 ci x newr) i j ==
           (if i = ci then acc1 newr j else acc2 x i j))
  = ()
#pop-options

(* Row [ci] of [chest_update_slice 0 ci x newr] is [newr]. *)
let ematrix_row_upd_slice_self (#et:Type) (#r #cc:nat)
  (x:chest2 et r cc) (ci:natlt r) (newr:chest1 et cc)
  : Lemma (ematrix_row (chest_update_slice 0 ci x newr) ci == chest1_to_seq newr)
  = Classical.forall_intro (acc2_chest_update_slice0 x ci newr ci);
    Seq.lemma_eq_intro
      (ematrix_row (chest_update_slice 0 ci x newr) ci) (chest1_to_seq newr)

(* Other rows of [chest_update_slice 0 ci x newr] are unchanged. *)
let ematrix_row_upd_slice_other (#et:Type) (#r #cc:nat)
  (x:chest2 et r cc) (ci:natlt r) (newr:chest1 et cc) (k:natlt r{k <> ci})
  : Lemma (ematrix_row (chest_update_slice 0 ci x newr) k == ematrix_row x k)
  = Classical.forall_intro (acc2_chest_update_slice0 x ci newr k);
    Seq.lemma_eq_intro
      (ematrix_row (chest_update_slice 0 ci x newr) k) (ematrix_row x k)

(* From the CPU, read one element of a flat rank-1 [array1] under [gpu_loc].
   Direct replacement for the deleted [Kuiper.Array1.arr_read_1]: copies the
   single device element into a length-1 host [vec] via the sanctioned
   [gpu_memcpy_device_to_host'] primitive, reads it, and restores the tensor
   view.  Specialised to [f32] to keep instance resolution simple. *)
inline_for_extraction noextract
fn t_read_1
  (#len : erased nat)
  (a : array1 f32 (l1_forward len))
  (i : szlt len)
  (#f : perm)
  (#va : chest1 f32 len)
  preserves cpu
  preserves on gpu_loc (a |-> Frac f va)
  returns x : f32
  ensures pure (x == acc1 va (SZ.v i))
{
  let ca = Pulse.Lib.Vec.alloc #f32 (zero #f32) 1sz;
  map_loc gpu_loc
    #(a |-> Frac f va)
    #(core a |-> Frac f (to_seq (l1_forward len) va))
    fn _ { tensor_concr a; };
  gpu_memcpy_device_to_host' #_ #_ #(hide 1) ca 0sz (core a) i 1sz;
  let x = Pulse.Lib.Vec.(ca.(0sz));
  Pulse.Lib.Vec.free ca;
  map_loc gpu_loc
    #(core a |-> Frac f (to_seq (l1_forward len) va))
    #(a |-> Frac f va)
    fn _ {
      tensor_abs (l1_forward len) (core a);
      rewrite (from_array (l1_forward len) (core a) |-> Frac f va)
           as (a |-> Frac f va);
    };
  lem_to_seq len va;
  x
}

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

let ematrix_upd_row_self
  (#et : Type0)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  (new_row : lseq et cols)
  : Lemma (ematrix_row (ematrix_upd_row em i new_row) i == new_row)
          [SMTPat (ematrix_row (ematrix_upd_row em i new_row) i)]
  = Seq.lemma_eq_intro
      (ematrix_row (ematrix_upd_row em i new_row) i)
      new_row

let ematrix_upd_row_other
  (#et : Type0)
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  (new_row : lseq et cols)
  (k : natlt rows)
  : Lemma (requires k <> i)
          (ensures ematrix_row (ematrix_upd_row em i new_row) k ==
                   ematrix_row em k)
          [SMTPat (ematrix_row (ematrix_upd_row em i new_row) k)]
  = Seq.lemma_eq_intro
      (ematrix_row (ematrix_upd_row em i new_row) k)
      (ematrix_row em k)

(* Algebraic identity: applying [affine_step inv neg_mean_inv] then
   [affine_step γ β] equals [bn_step inv neg_mean_inv γ β].  Both sides
   unfold to [add (mul (add (mul x inv) neg_mean_inv) γ) β]; this is a
   one-line definitional equality, lifted to the whole row by
   [Seq.lemma_eq_intro]. *)
let bn_via_double_affine_lemma
  (#nhw : nat)
  (inv neg_mean_inv g b : f32)
  (row : Seq.lseq f32 nhw)
  : Lemma
      (affine_result g b
         (affine_result inv neg_mean_inv row)
       == bn_row_result inv neg_mean_inv g b row)
  = let lhs : Seq.lseq f32 nhw =
      affine_result g b
        (affine_result inv neg_mean_inv row) in
    let rhs : Seq.lseq f32 nhw =
      bn_row_result inv neg_mean_inv g b row in
    let aux (k : nat { k < nhw })
      : Lemma (Seq.index lhs k == Seq.index rhs k)
      = () in
    Classical.forall_intro aux;
    Seq.lemma_eq_intro lhs rhs

let bn_row_result_transport
  (#nhw : nat)
  (out row row' : Seq.lseq f32 nhw)
  (inv neg_mean_inv g g' b b' : f32)
  : Lemma
      (requires
        out == bn_row_result inv neg_mean_inv g b row /\
        row == row' /\ g == g' /\ b == b')
      (ensures out == bn_row_result inv neg_mean_inv g' b' row')
  = ()

#push-options "--z3rlimit 30 --fuel 2 --ifuel 2"
let row_batch_normalized_stable
  (#nhw c_dim : nat)
  (sx sx_old sx_new : chest2 f32 c_dim nhw)
  (gamma beta : Seq.lseq f32 c_dim)
  (ci : nat { ci < c_dim })
  (eps inv_n : f32)
  : Lemma (requires
            row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
              sx sx_old gamma beta ci eps inv_n /\
            ematrix_row sx_new ci == ematrix_row sx_old ci)
          (ensures
            row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
              sx sx_new gamma beta ci eps inv_n)
  = ()
#pop-options


#push-options "--z3rlimit 30 --fuel 2 --ifuel 2"
let row_batch_normalized_intro
  (#nhw c_dim : nat)
  (sx sx' : chest2 f32 c_dim nhw)
  (gamma beta : Seq.lseq f32 c_dim)
  (ci : nat { ci < c_dim })
  (eps inv_n : f32)
  (sum sumsq mean m2 var var_eps inv neg_mean_inv : f32)
  : Lemma
      (requires
        (let row : Seq.lseq f32 nhw = ematrix_row sx ci in
         sum   %~ rsum (to_real_seq row) /\
         sumsq %~ frobenius_sumsq_r (to_real_seq row) /\
         mean         == mul sum   inv_n /\
         m2           == mul sumsq inv_n /\
         var          == sub m2 (mul mean mean) /\
         var_eps      == Kuiper.Scalars.add var eps /\
         inv          == rsqrt var_eps /\
         neg_mean_inv == sub (zero #f32)
                           (mul mean inv) /\
         ematrix_row sx' ci ==
           bn_row_result inv neg_mean_inv
             (Seq.index gamma ci) (Seq.index beta ci) row))
      (ensures
        row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
          sx sx' gamma beta ci eps inv_n)
  = ()
#pop-options

#push-options "--z3rlimit 30 --fuel 2 --ifuel 2"
let row_batch_normalized_change_sx
  (#nhw c_dim : nat)
  (sx_a sx_b sx' : chest2 f32 c_dim nhw)
  (gamma beta : Seq.lseq f32 c_dim)
  (ci : nat { ci < c_dim })
  (eps inv_n : f32)
  : Lemma
      (requires
        row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
          sx_a sx' gamma beta ci eps inv_n /\
        ematrix_row sx_a ci == ematrix_row sx_b ci)
      (ensures
        row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
          sx_b sx' gamma beta ci eps inv_n)
  = ()
#pop-options

let row_batch_normalized_stable_forall
  (#nhw c_dim : nat)
  (sx sx_old sx_new : chest2 f32 c_dim nhw)
  (gamma beta : Seq.lseq f32 c_dim)
  (vi : nat { vi <= c_dim })
  (eps inv_n : f32)
  : Lemma
      (requires
        (forall (ci : nat). ci < vi ==>
          row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
            sx sx_old gamma beta ci eps inv_n) /\
        (forall (k : nat). k < c_dim /\ k <> vi ==>
          ematrix_row sx_new k == ematrix_row sx_old k))
      (ensures
        forall (ci : nat). ci < vi ==>
          row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
            sx sx_new gamma beta ci eps inv_n)
  = introduce forall (ci : nat). ci < vi ==>
      row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
        sx sx_new gamma beta ci eps inv_n
    with introduce _ ==> _
    with (
      assert (ci < c_dim);
      assert (ci <> vi);
      assert (ematrix_row sx_new ci == ematrix_row sx_old ci);
      row_batch_normalized_stable #nhw c_dim
        sx sx_old sx_new gamma beta ci eps inv_n
    )

let row_batch_normalized_extend_forall
  (#nhw c_dim : nat)
  (sx sx_new : chest2 f32 c_dim nhw)
  (gamma beta : Seq.lseq f32 c_dim)
  (vi : nat { vi < c_dim })
  (eps inv_n : f32)
  : Lemma
      (requires
        (forall (ci : nat). ci < vi ==>
          row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
            sx sx_new gamma beta ci eps inv_n) /\
        row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
          sx sx_new gamma beta vi eps inv_n)
      (ensures
        forall (ci : nat). ci < vi + 1 ==>
          row_batch_normalized #f32 #_ #_ #_ #c_dim #nhw
            sx sx_new gamma beta ci eps inv_n)
  = ()

(* Per-channel body: extract row, two reductions for sum and sumsq, two
   in-place affines (one for (x-μ)/σ, one for γ·+β), restore row. *)
#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn batchnorm_channel
  (n  : erased nat)
  (c  : szp)
  (hw : szp { SZ.v hw > 0 /\
              SZ.fits (n * SZ.v hw) /\
              SZ.fits (SZ.v hw * SZ.v c) /\
              SZ.fits (n * (SZ.v hw * SZ.v c)) })
  (nhw : szp { SZ.v nhw == n * SZ.v hw /\
               nhw <= max_blocks * max_threads /\
               SZ.fits (SZ.v nhw + 1024) })
  (ci : szlt c)
  (eps inv_n : f32)
  (x : array2 f32 (l2_bcm_channels n (SZ.v c) (SZ.v hw))
                   { is_global x })
  (gamma : array1 f32 (l1_forward c) { is_global gamma })
  (beta  : array1 f32 (l1_forward c) { is_global beta  })
  (#fg #fb : perm)
  (#sx : chest2 f32 (SZ.v c) (n * SZ.v hw))
  (#sg : chest1 f32 (SZ.v c))
  (#sb : chest1 f32 (SZ.v c))
  preserves cpu
  preserves
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sb)
  requires
    on gpu_loc (x |-> sx)
  ensures
    (exists* (sx' : chest2 f32 (SZ.v c) (n * SZ.v hw)).
       on gpu_loc (x |-> sx') **
       pure (row_batch_normalized #f32 #_ #_ #_ #(SZ.v c) #(n * SZ.v hw)
               sx sx' (chest1_to_seq sg) (chest1_to_seq sb) (SZ.v ci) eps inv_n) **
       pure (forall (k : nat). k < SZ.v c /\ k <> SZ.v ci ==>
                ematrix_row sx' k == ematrix_row sx k))
{
  (* Read γ_c, β_c as host scalars (preserves the gpu permission). *)
  let g_c = t_read_1 gamma ci;
  let b_c = t_read_1 beta  ci;

  (* Split the row off the matrix while keeping the [on gpu_loc]: we now own
     [sliceof x 0 ci |-> chest_slice 0 ci sx] plus a restore wand. *)
  map_loc gpu_loc (fun () -> tensor_extract_slice x 0 (SZ.v ci));

  (* [chest1] view of the row (== [chest_slice 0 ci sx]) and its real image,
     shared by both reductions. *)
  let row_c : chest1 f32 (n * SZ.v hw) =
    hide (chest2_row (reveal sx) (SZ.v ci));
  let row_r : chest1 real (n * SZ.v hw) =
    hide (to_real_chest (reveal row_c));
  assert pure (reveal row_c %~ reveal row_r);

  (* The row as a flat [ematrix_row], for the functional-correctness spec. *)
  chest2_row_to_seq (reveal sx) (SZ.v ci);
  let row_g : erased (Seq.lseq f32 (n * SZ.v hw)) =
    hide (ematrix_row (reveal sx) (SZ.v ci));
  assert pure (chest1_to_seq (reveal row_c) == reveal row_g);

  (* sum = Σ row  (identity pre-map; reduce preserves the slice). *)
  let sum =
    HRed.reduce #f32 id id 1024sz nhw
      #_ #(c_bcm_channel_slice n c hw ci)
      (sliceof x 0 (SZ.v ci)) #row_c row_r;
  chest_map_to_seq (id #real) (reveal row_r);
  seq_map_id_eq #real (chest1_to_seq (reveal row_r));
  to_real_chest_to_seq (reveal row_c);
  assert pure (sum %~ rsum (to_real_seq (reveal row_g)));

  (* sumsq = Σ row²  (square pre-map). *)
  sq_step_approx_forall #f32 ();
  let sumsq =
    HRed.reduce #f32
      (square #f32) sq_step_r 1024sz nhw
      #_ #(c_bcm_channel_slice n c hw ci)
      (sliceof x 0 (SZ.v ci)) #row_c row_r;
  chest_map_to_seq sq_step_r (reveal row_r);
  assert pure (sumsq %~ frobenius_sumsq_r (to_real_seq (reveal row_g)));

  (* Per-channel scalars. *)
  let mean = mul sum inv_n;
  let m2 = mul sumsq inv_n;
  let var = sub m2 (mul mean mean);
  let var_eps = add var eps;
  let inv = rsqrt var_eps;
  let neg_mean_inv = sub (zero #f32) (mul mean inv);

  (* Pass 1: row ← (row - μ) * inv = inv*row + neg_mean_inv. *)
  Map.map_gpu (affine_step inv neg_mean_inv) nhw
    #_ #(c_bcm_channel_slice n c hw ci) (sliceof x 0 (SZ.v ci));

  (* Pass 2: row ← γ_c*row + β_c. *)
  Map.map_gpu (affine_step g_c b_c) nhw
    #_ #(c_bcm_channel_slice n c hw ci) (sliceof x 0 (SZ.v ci));

  (* Name the resulting slice chest [eC]. *)
  let eC : chest1 f32 (n * SZ.v hw) =
    hide (chest_map (affine_step g_c b_c)
            (chest_map (affine_step inv neg_mean_inv) (reveal row_c)));
  assert (on gpu_loc (sliceof x 0 (SZ.v ci) |-> reveal eC));

  (* Bridge [chest1_to_seq eC] to [bn_row_result]. *)
  chest_map_to_seq (affine_step inv neg_mean_inv) (reveal row_c);
  chest_map_to_seq (affine_step g_c b_c)
    (chest_map (affine_step inv neg_mean_inv) (reveal row_c));
  bn_via_double_affine_lemma #(n * SZ.v hw)
    inv neg_mean_inv g_c b_c (reveal row_g);
  assert pure (chest1_to_seq (reveal eC) ==
    affine_result g_c b_c
      (affine_result inv neg_mean_inv (reveal row_g)));
  assert pure (chest1_to_seq (reveal eC) ==
    bn_row_result inv neg_mean_inv g_c b_c (reveal row_g));

  (* Recombine into the full matrix by applying the restore wand at [eC]. *)
  assert pure (modulo_i 0 (SZ.v c @| (n * SZ.v hw) @| INil) ==
               (n * SZ.v hw) @| INil);
  map_loc gpu_loc
    #(sliceof x 0 (SZ.v ci) |-> reveal eC **
      (forall* (s' : chest (modulo_i 0 (SZ.v c @| (n * SZ.v hw) @| INil)) f32).
        sliceof x 0 (SZ.v ci) |-> s' @==>
        x |-> chest_update_slice 0 (SZ.v ci) (reveal sx) s'))
    #(x |-> chest_update_slice 0 (SZ.v ci) (reveal sx) (reveal eC))
    fn () {
      elim_forall
        (reveal eC <: chest (modulo_i 0 (SZ.v c @| (n * SZ.v hw) @| INil)) f32);
      elim_trade _ _;
      ()
    };

  (* [sx_final] is [sx] with row [ci] replaced by [eC]. *)
  let sx_final : chest2 f32 (SZ.v c) (n * SZ.v hw) =
    hide (chest_update_slice 0 (SZ.v ci) (reveal sx) (reveal eC));
  assert (on gpu_loc (x |-> reveal sx_final));

  (* Row [ci] of [sx_final] is [eC == bn_row_result]; other rows unchanged. *)
  ematrix_row_upd_slice_self (reveal sx) (SZ.v ci) (reveal eC);
  Classical.forall_intro
    (ematrix_row_upd_slice_other (reveal sx) (SZ.v ci) (reveal eC));
  assert pure (ematrix_row (reveal sx_final) (SZ.v ci) ==
    bn_row_result inv neg_mean_inv g_c b_c (reveal row_g));
  acc1_chest1_to_seq (reveal sg) (SZ.v ci);
  acc1_chest1_to_seq (reveal sb) (SZ.v ci);
  assert pure (g_c == Seq.index (chest1_to_seq (reveal sg)) (SZ.v ci));
  assert pure (b_c == Seq.index (chest1_to_seq (reveal sb)) (SZ.v ci));
  bn_row_result_transport
    (ematrix_row (reveal sx_final) (SZ.v ci)) (reveal row_g)
    (ematrix_row (reveal sx) (SZ.v ci)) inv neg_mean_inv g_c
    (Seq.index (chest1_to_seq (reveal sg)) (SZ.v ci)) b_c
    (Seq.index (chest1_to_seq (reveal sb)) (SZ.v ci));
  assert pure (
    forall (k : nat). k < SZ.v c /\ k <> SZ.v ci ==>
      ematrix_row (reveal sx_final) k == ematrix_row sx k);
  row_batch_normalized_intro #(n * SZ.v hw) (SZ.v c)
    (reveal sx) (reveal sx_final)
    (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sb))
    (SZ.v ci) eps inv_n
    sum sumsq mean m2 var var_eps inv neg_mean_inv;
  assert pure (row_batch_normalized #f32 #_ #_ #_ #(SZ.v c) #(n * SZ.v hw)
    (reveal sx) (reveal sx_final)
    (chest1_to_seq (reveal sg)) (chest1_to_seq (reveal sb))
    (SZ.v ci) eps inv_n);
  ()
}
#pop-options

(* Whole-tensor entry point: loop over channels. *)
#push-options "--z3rlimit 5"
inline_for_extraction noextract
fn batch_norm
  (n  : erased nat)
  (c  : szp)
  (hw : szp { SZ.v hw > 0 /\
              SZ.fits (n * SZ.v hw) /\
              SZ.fits (SZ.v hw * SZ.v c) /\
              SZ.fits (n * (SZ.v hw * SZ.v c)) })
  (nhw : szp { SZ.v nhw == n * SZ.v hw /\
               nhw <= max_blocks * max_threads /\
               SZ.fits (SZ.v nhw + 1024) })
  (eps inv_n : f32)
  (x : array2 f32 (l2_bcm_channels n (SZ.v c) (SZ.v hw))
                   { is_global x })
  (gamma : array1 f32 (l1_forward c) { is_global gamma })
  (beta  : array1 f32 (l1_forward c) { is_global beta  })
  (#fg #fb : perm)
  (#sx : chest2 f32 (SZ.v c) (n * SZ.v hw))
  (#sg : chest1 f32 (SZ.v c))
  (#sb : chest1 f32 (SZ.v c))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sb)
  ensures
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sb) **
    (exists* (sx' : chest2 f32 (SZ.v c) (n * SZ.v hw)).
       on gpu_loc (x |-> sx') **
       pure (batchnorm_post (SZ.v c) (n * SZ.v hw) eps inv_n
               (chest1_to_seq sg) (chest1_to_seq sb) sx sx'))
{
  (* Work with the flat [lseq] views of γ/β for the spec-level predicates
     (which take [Seq.lseq]); the gpu permissions keep the [chest1] values. *)
  let sg : erased (lseq f32 (SZ.v c)) = hide (chest1_to_seq (reveal sg));
  let sb : erased (lseq f32 (SZ.v c)) = hide (chest1_to_seq (reveal sb));
  let mut idx = 0sz;
  while (let i = !idx; SZ.(i <^ c))
    invariant
      exists* (vi : sz) (sx' : chest2 f32 (SZ.v c) (n * SZ.v hw)).
        idx |-> vi **
        on gpu_loc (x |-> sx') **
        cpu **
        pure (SZ.v vi <= SZ.v c /\
              (forall (ci : nat). ci < SZ.v vi ==>
                 row_batch_normalized #f32 #_ #_ #_ #(SZ.v c) #(n * SZ.v hw)
                   sx sx' sg sb ci eps inv_n) /\
              (forall (ci : nat). SZ.v vi <= ci /\ ci < SZ.v c ==>
                 ematrix_row sx' ci == ematrix_row sx ci))
    decreases (SZ.v c - SZ.v !idx)
  {
    let i = !idx;
    with sx'_pre. assert (on gpu_loc (x |-> sx'_pre));
    assert pure (ematrix_row sx'_pre (SZ.v i) == ematrix_row sx (SZ.v i));
    batchnorm_channel n c hw nhw i eps inv_n x gamma beta;
    with sx'_new. assert (on gpu_loc (x |-> sx'_new));
    assert pure (
      forall (k : nat). k < SZ.v c /\ k <> SZ.v i ==>
        ematrix_row sx'_new k == ematrix_row sx'_pre k);
    row_batch_normalized_stable_forall #(n * SZ.v hw) (SZ.v c)
      sx sx'_pre sx'_new sg sb (SZ.v i) eps inv_n;
    assert pure (
      forall (ci : nat). ci < SZ.v i ==>
        row_batch_normalized #f32 #_ #_ #_ #(SZ.v c) #(n * SZ.v hw)
          sx sx'_new sg sb ci eps inv_n);
    assert pure (
      row_batch_normalized #f32 #_ #_ #_ #(SZ.v c) #(n * SZ.v hw)
        sx'_pre sx'_new sg sb (SZ.v i) eps inv_n);
    row_batch_normalized_change_sx #(n * SZ.v hw) (SZ.v c)
      sx'_pre sx sx'_new sg sb (SZ.v i) eps inv_n;
    assert pure (
      row_batch_normalized #f32 #_ #_ #_ #(SZ.v c) #(n * SZ.v hw)
        sx sx'_new sg sb (SZ.v i) eps inv_n);
    row_batch_normalized_extend_forall #(n * SZ.v hw) (SZ.v c)
      sx sx'_new sg sb (SZ.v i) eps inv_n;
    assert pure (
      forall (ci : nat). SZ.v i + 1 <= ci /\ ci < SZ.v c ==>
        ematrix_row sx'_new ci == ematrix_row sx ci);
    assert pure (
      forall (ci : nat). ci < SZ.v i + 1 ==>
        row_batch_normalized #f32 #_ #_ #_ #(SZ.v c) #(n * SZ.v hw)
          sx sx'_new sg sb ci eps inv_n);
    idx := SZ.(!idx +^ 1sz);
  };
  ()
}
#pop-options

(* Public entry point: compute the per-channel reciprocal [bn_inv_n nhw]
   inside the verification boundary, then delegate to [batch_norm].  The
   heavy proofs above treat [inv_n] abstractly, so the constant
   computation here does not affect their cost. *)
fn batchnorm_fw
  (n  : erased nat)
  (c  : szp)
  (hw : szp { SZ.v hw > 0 /\
              SZ.fits (n * SZ.v hw) /\
              SZ.fits (SZ.v hw * SZ.v c) /\
              SZ.fits (n * (SZ.v hw * SZ.v c)) })
  (nhw : szp { SZ.v nhw == n * SZ.v hw /\
               nhw <= max_blocks * max_threads /\
               SZ.fits (SZ.v nhw + 1024) })
  (eps : f32)
  (x : array2 f32 (l2_bcm_channels n (SZ.v c) (SZ.v hw))
                   { is_global x })
  (gamma : array1 f32 (l1_forward c) { is_global gamma })
  (beta  : array1 f32 (l1_forward c) { is_global beta  })
  (#fg #fb : perm)
  (#sx : chest2 f32 (SZ.v c) (n * SZ.v hw))
  (#sg : chest1 f32 (SZ.v c))
  (#sb : chest1 f32 (SZ.v c))
  preserves cpu
  requires
    on gpu_loc (x |-> sx) **
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sb)
  ensures
    on gpu_loc (gamma |-> Frac fg sg) **
    on gpu_loc (beta  |-> Frac fb sb) **
    (exists* (sx' : chest2 f32 (SZ.v c) (n * SZ.v hw)).
       on gpu_loc (x |-> sx') **
       pure (batchnorm_post (SZ.v c) (n * SZ.v hw) eps (bn_inv_n nhw)
               (chest1_to_seq sg) (chest1_to_seq sb) sx sx'))
{
  let inv_n : f32 = bn_inv_n nhw;
  batch_norm n c hw nhw eps inv_n x gamma beta;
}

let batchnorm_fw_f32 : batchnorm_fw_ty f32 = batchnorm_fw
