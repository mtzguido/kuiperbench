module Kuiper.Kernel.TriangularMatmul

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Tensor
open Kuiper.Tensor { tensor_pts_to_cell as pts_to_cell }
open Kuiper.Tensor.Layout.Alg { l2_row_major, l2_col_major }
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Bijection
open Kuiper.EMatrix { mtranspose }
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module TGT = Kuiper.Ghost.TensorTranspose

let abs_bij (#m #n : nat)
  : (abs (m @| n @| INil) =~ (natlt m & natlt n)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
  }

let term
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  (k : natlt n)
  : real
  = mul (acc2 a i k) (acc2 b k j)

#push-options "--z3rlimit 20"
let rec prefix_zero
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  (to : nat { to <= n })
  (hzero : squash (forall (k : natlt n). k < to ==> term a b i j k == 0.0R))
  : Lemma
      (ensures MS.__matmul_single a b i j to == 0.0R)
      (decreases to)
  = if to = 0 then
      MS.matmul_zero_lemma a b i j
    else begin
      prefix_zero a b i j (to - 1) ();
      MS.matmul_single_lemma a b i j to
    end
#pop-options

#push-options "--z3rlimit 20"
let rec suffix_zero
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  (from to : nat { from <= to /\ to <= n })
  (hzero : squash (forall (k : natlt n). from <= k /\ k < to ==>
                                      term a b i j k == 0.0R))
  : Lemma
      (ensures
        MS.__matmul_single a b i j to ==
        MS.__matmul_single a b i j from)
      (decreases to - from)
  = if to > from then begin
      suffix_zero a b i j from (to - 1) ();
      MS.matmul_single_lemma a b i j to
    end
#pop-options

let upper_prefix_zero
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  : Lemma
      (requires is_upper_triangular a)
      (ensures MS.__matmul_single a b i j i == 0.0R)
  = prefix_zero a b i j i ()

let upper_suffix_zero
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  : Lemma
      (requires is_upper_triangular b)
      (ensures
        MS.__matmul_single a b i j n ==
        MS.__matmul_single a b i j (j + 1))
  = suffix_zero a b i j (j + 1) n ()

let upper_matmul_cell_zero
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  : Lemma
      (requires
        is_upper_triangular a /\ is_upper_triangular b /\ j < i)
      (ensures acc2 (MS.matmul a b) i j == 0.0R)
  =
  upper_prefix_zero a b i j;
  suffix_zero a b i j i n ();
  MS.lemma_matmul_index a b i j

let lower_matmul_cell_zero
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  : Lemma
      (requires
        is_lower_triangular a /\ is_lower_triangular b /\ i < j)
      (ensures acc2 (MS.matmul a b) i j == 0.0R)
  =
  prefix_zero a b i j j ();
  suffix_zero a b i j j n ();
  MS.lemma_matmul_index a b i j

let upper_matmul_cell_prop
  (#n : nat)
  (a b : chest2 real n n)
  (htriangular : squash (
    is_upper_triangular a /\ is_upper_triangular b))
  (i j : natlt n)
  : Lemma
      (ensures j < i ==> acc2 (MS.matmul a b) i j == zero)
  = if j < i then upper_matmul_cell_zero a b i j

let lower_matmul_cell_prop
  (#n : nat)
  (a b : chest2 real n n)
  (htriangular : squash (
    is_lower_triangular a /\ is_lower_triangular b))
  (i j : natlt n)
  : Lemma
      (ensures i < j ==> acc2 (MS.matmul a b) i j == zero)
  = if i < j then lower_matmul_cell_zero a b i j

let upper_matmul_is_upper
  (#n : nat)
  (a b : chest2 real n n)
  : Lemma
      (requires is_upper_triangular a /\ is_upper_triangular b)
      (ensures is_upper_triangular (MS.matmul a b))
  = Classical.forall_intro_2 (upper_matmul_cell_prop a b ())

let lower_matmul_is_lower
  (#n : nat)
  (a b : chest2 real n n)
  : Lemma
      (requires is_lower_triangular a /\ is_lower_triangular b)
      (ensures is_lower_triangular (MS.matmul a b))
  = Classical.forall_intro_2 (lower_matmul_cell_prop a b ())

let upper_partial_is_matmul
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n { i <= j })
  : Lemma
      (requires is_upper_triangular a /\ is_upper_triangular b)
      (ensures
        MS.__matmul_single a b i j (j + 1) ==
        acc2 (MS.matmul a b) i j)
  =
  upper_suffix_zero a b i j;
  MS.lemma_matmul_index a b i j

let upper_reduction_is_matmul
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  (to : nat {
    (i <= j /\ to == j + 1) \/
    (j < i /\ to == i) })
  : Lemma
      (requires is_upper_triangular a /\ is_upper_triangular b)
      (ensures
        MS.__matmul_single a b i j to ==
        acc2 (MS.matmul a b) i j)
  = if i <= j then
      upper_partial_is_matmul a b i j
    else begin
      upper_prefix_zero a b i j;
      upper_matmul_cell_zero a b i j
    end

unfold
let kpre
  (#et : Type0) {| floating et, real_like et |}
  (n : szp)
  (#lA #lB #lC : layout2 n n)
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (sA sB sC : chest2 et n n)
  (rA : chest2 real n n { sA %~ rA /\ is_upper_triangular rA })
  (rB : chest2 real n n { sB %~ rB /\ is_upper_triangular rB })
  (fA fB : perm)
  (gid : natlt (n * n))
  : slprop
  = gA |-> Frac (fA /. (n * n)) sA **
    gB |-> Frac (fB /. (n * n)) sB **
    pts_to_cell gC
      (gid / n, (gid % n, ()))
      (acc sC (gid / n, (gid % n, ())))

unfold
let kpost
  (#et : Type0) {| floating et, real_like et |}
  (n : szp)
  (#lA #lB #lC : layout2 n n)
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (sA sB : chest2 et n n)
  (rA : chest2 real n n { sA %~ rA /\ is_upper_triangular rA })
  (rB : chest2 real n n { sB %~ rB /\ is_upper_triangular rB })
  (fA fB : perm)
  (gid : natlt (n * n))
  : slprop
  = gA |-> Frac (fA /. (n * n)) sA **
    gB |-> Frac (fB /. (n * n)) sB **
    exists* v.
      pts_to_cell gC (gid / n, (gid % n, ())) v **
      pure (v %~ acc2 (MS.matmul rA rB) (gid / n) (gid % n))

inline_for_extraction noextract
fn kf
  (#et : Type0) {| floating et, real_like et |}
  (n : szp)
  (#lA #lB #lC : layout2 n n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#sA #sB #sC : chest2 et n n)
  (#rA : chest2 real n n { sA %~ rA /\ is_upper_triangular rA })
  (#rB : chest2 real n n { sB %~ rB /\ is_upper_triangular rB })
  (#fA #fB : perm)
  (gid : szlt (n *^ n))
  ()
  norewrite
  requires
    gpu ** kpre n gA gB gC sA sB sC rA rB fA fB gid
  ensures
    gpu ** kpost n gA gB gC sA sB rA rB fA fB gid
{
  let row : szlt n = gid /^ n; assert rewrites_to row (gid /^ n);
  let col : szlt n = gid %^ n; assert rewrites_to col (gid %^ n);

  upper_prefix_zero rA rB (SZ.v row) (SZ.v col);

  let mut acc : et = zero;
  let mut k : sz = row;
  while (!k <=^ col)
    invariant
      live acc ** live k **
      gA |-> Frac (fA /. (n * n)) sA **
      gB |-> Frac (fB /. (n * n)) sB **
      pts_to_cell gC (SZ.v row, (SZ.v col, ()))
        (acc2 sC (SZ.v row) (SZ.v col)) **
      pure (
        SZ.v row <= SZ.v !k /\ SZ.v !k <= SZ.v n /\
        (SZ.v row <= SZ.v col ==> SZ.v !k <= SZ.v col + 1) /\
        (SZ.v col < SZ.v row ==> SZ.v !k == SZ.v row) /\
        !acc %~ MS.__matmul_single rA rB
          (SZ.v row) (SZ.v col) (SZ.v !k)) **
      emp
    decreases (col + 1 - SZ.v !k)
  {
    let kk_raw : sz = !k;
    assert pure (SZ.v kk_raw < SZ.v n);
    let kk : szlt n = kk_raw;
    let va = tensor_read gA (row, (kk, ()));
    let vb = tensor_read gB (kk, (col, ()));
    let prod = va `mul` vb;
    a_mul va vb
      (acc2 rA (SZ.v row) (SZ.v kk))
      (acc2 rB (SZ.v kk) (SZ.v col));
    a_add !acc prod
      (MS.__matmul_single rA rB (SZ.v row) (SZ.v col) (SZ.v kk))
      (term rA rB (SZ.v row) (SZ.v col) (SZ.v kk));
    MS.matmul_single_lemma rA rB
      (SZ.v row) (SZ.v col) (SZ.v kk + 1);
    acc := !acc `add` prod;
    k := !k +^ 1sz;
  };

  upper_reduction_is_matmul rA rB
    (SZ.v row) (SZ.v col) (SZ.v !k);
  tensor_write_cell gC (row, (col, ())) !acc;
}

ghost
fn setup
  (#et : Type0) {| floating et, real_like et |}
  (n : szp)
  (#lA #lB #lC : layout2 n n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#sA #sB #sC : chest2 et n n)
  (#rA : chest2 real n n { sA %~ rA /\ is_upper_triangular rA })
  (#rB : chest2 real n n { sB %~ rB /\ is_upper_triangular rB })
  (fA fB : perm)
  ()
  norewrite
  requires
    gA |-> Frac fA sA **
    gB |-> Frac fB sB **
    gC |-> sC
  ensures
    (forall+ (gid : natlt (n *^ n)).
      kpre n gA gB gC sA sB sC rA rB fA fB gid) ** emp
{
  tensor_share_n gA (n *^ n);
  tensor_share_n gB (n *^ n);

  tensor_explode gC;
  forevery_iso (abs_bij #n #n) _;
  forevery_ext _ (fun (ij : natlt n & natlt n) ->
    pts_to_cell gC (fst ij, (snd ij, ()))
      (acc sC (fst ij, (snd ij, ()))));
  forevery_unflatten' _;
  forevery_unfactor' (n *^ n) n n (fun r c ->
    pts_to_cell gC (r, (c, ())) (acc sC (r, (c, ()))));

  forevery_zip3 #(natlt2 n n)
    (fun _ -> gA |-> Frac (fA /. (n *^ n)) sA)
    (fun _ -> gB |-> Frac (fB /. (n *^ n)) sB)
    (fun i -> pts_to_cell gC
      ((i / n <: natlt n), ((i % n <: natlt n), ()))
      (acc sC ((i / n <: natlt n), ((i % n <: natlt n), ()))));
  forevery_ext #(natlt2 n n) _
    (kpre n gA gB gC sA sB sC rA rB fA fB);
}

ghost
fn teardown
  (#et : Type0) {| floating et, real_like et |}
  (n : szp)
  (#lA #lB #lC : layout2 n n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA)
  (gB : array2 et lB)
  (gC : array2 et lC)
  (#sA #sB : chest2 et n n)
  (#rA : chest2 real n n { sA %~ rA /\ is_upper_triangular rA })
  (#rB : chest2 real n n { sB %~ rB /\ is_upper_triangular rB })
  (fA fB : perm)
  ()
  norewrite
  requires
    (forall+ (gid : natlt (n *^ n)).
      kpost n gA gB gC sA sB rA rB fA fB gid) ** emp
  ensures
    gA |-> Frac fA sA **
    gB |-> Frac fB sB **
    (exists* (sC' : chest2 et n n).
      gC |-> sC' ** pure (sC' %~ MS.matmul rA rB))
{
  forevery_unzip3
    (fun (_ : natlt (n *^ n)) -> gA |-> Frac (fA /. (n * n)) sA)
    (fun (_ : natlt (n *^ n)) -> gB |-> Frac (fB /. (n * n)) sB) _;

  forevery_rw_type (natlt (n *^ n)) (natlt (n * n))
    (fun _ -> gA |-> Frac (fA /. (n * n)) sA);
  forevery_rw_type (natlt (n *^ n)) (natlt (n * n))
    (fun _ -> gB |-> Frac (fB /. (n * n)) sB);
  tensor_gather_n gA _;
  tensor_gather_n gB _;

  forevery_factor (n *^ n) n n _;
  let vf = forevery_exists_2 #(natlt n) #_ #(natlt n) _;
  assert (pure (forall (r c : nat). c < n ==> (r * n + c) / n == r));
  assert (pure (forall (r c : nat). c < n ==> (r * n + c) % n == c));
  forevery_ext_2 _
    (fun (r : natlt n) (c : natlt n) ->
      pts_to_cell gC (r, (c, ())) (vf r c) **
      pure (vf r c %~ acc2 (MS.matmul rA rB) r c));

  forevery_extract_pure_2
    (fun (r : natlt n) (c : natlt n) ->
      pts_to_cell gC (r, (c, ())) (vf r c) **
      pure (vf r c %~ acc2 (MS.matmul rA rB) r c))
    (fun (r : natlt n) (c : natlt n) ->
      vf r c %~ acc2 (MS.matmul rA rB) r c)
    fn r c { (); };

  let sC' : chest2 et n n = mk2 (fun r c -> vf r c);
  ghost
  fn aux (r : natlt n) (c : natlt n)
    requires
      pts_to_cell gC (r, (c, ())) (vf r c) **
      pure (vf r c %~ acc2 (MS.matmul rA rB) r c)
    ensures pts_to_cell gC (r, (c, ())) (acc2 sC' r c)
  {
    assert pure (acc2 sC' r c == vf r c);
  };
  forevery_map_2 #(natlt n) #(natlt n)
    (fun r c ->
      pts_to_cell gC (r, (c, ())) (vf r c) **
      pure (vf r c %~ acc2 (MS.matmul rA rB) r c))
    (fun r c -> pts_to_cell gC (r, (c, ())) (acc2 sC' r c))
    aux;
  forevery_flatten' (fun (rc : natlt n & natlt n) ->
    pts_to_cell gC (fst rc, (snd rc, ())) (acc2 sC' (fst rc) (snd rc)));
  forevery_iso (bij_sym (abs_bij #n #n)) _;
  forevery_ext _ (fun (idx : abs (n @| n @| INil)) ->
    pts_to_cell gC idx (acc sC' idx));
  tensor_implode gC;
  assert pure (sC' %~ MS.matmul rA rB);
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| floating et, real_like et |}
  (n : szp { n * n <= max_blocks * max_threads })
  (#lA #lB #lC : layout2 n n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (#sA #sB #sC : chest2 et n n)
  (rA : chest2 real n n { sA %~ rA /\ is_upper_triangular rA })
  (rB : chest2 real n n { sB %~ rB /\ is_upper_triangular rB })
  (#fA #fB : perm)
  : kernel_desc
      (gA |-> Frac fA sA ** gB |-> Frac fB sB ** gC |-> sC)
      (gA |-> Frac fA sA ** gB |-> Frac fB sB **
       (exists* (sC' : chest2 et n n).
         gC |-> sC' ** pure (sC' %~ MS.matmul rA rB)))
  = {
      nthr = n *^ n;
      frame = emp;
      setup = setup n gA gB gC fA fB;
      teardown = teardown n gA gB gC fA fB;
      kpre = kpre n gA gB gC sA sB sC rA rB fA fB;
      kpost = kpost n gA gB gC sA sB rA rB fA fB;
      f = kf n gA gB gC;
      kpre_sendable = solve;
      kpost_sendable = solve;
    } <: kernel_desc_n _ _

inline_for_extraction noextract
fn upper_triangular_matmul
  (#et:Type0) {| floating et, real_like et |}
  (n : szp { n * n <= max_blocks * max_threads })
  (#lA : layout2 n n) {| ctlayout lA |}
  (gA : array2 et lA { is_global gA })
  (#lB : layout2 n n) {| ctlayout lB |}
  (gB : array2 et lB { is_global gB })
  (#lC : layout2 n n) {| ctlayout lC |}
  (gC : array2 et lC { is_global gC })
  (#sA #sB #sC : chest2 et n n)
  (#rA #rB : chest2 real n n)
  preserves
    cpu ** on gpu_loc (gA |-> sA) ** on gpu_loc (gB |-> sB)
  requires
    on gpu_loc (gC |-> sC) **
    pure (
      sA %~ rA /\ sB %~ rB /\
      is_upper_triangular rA /\ is_upper_triangular rB)
  ensures
    exists* (sC' : chest2 et n n).
      on gpu_loc (gC |-> sC') **
      pure (sC' %~ MS.matmul rA rB)
{
  launch_sync (kdesc n gA gB gC rA rB);
}

let mtranspose_involutive
  (#et : Type) (#n : nat) (m : chest2 et n n)
  : Lemma (ensures mtranspose (mtranspose m) == m)
  = assert (Kuiper.Chest.equal (mtranspose (mtranspose m)) m)

let mtranspose_approx
  (#et : Type0) {| scalar et, real_like et |}
  (#n : nat)
  (s : chest2 et n n)
  (r : chest2 real n n)
  : Lemma
      (requires s %~ r)
      (ensures mtranspose s %~ mtranspose r)
  = ()

let lower_transpose_is_upper
  (#t : Type0) {| scalar t |}
  (#n : nat)
  (a : chest2 t n n)
  : Lemma
      (requires is_lower_triangular a)
      (ensures is_upper_triangular (mtranspose a))
  = ()

#push-options "--z3rlimit 20"
let rec matmul_transpose_partial
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  (to : nat { to <= n })
  : Lemma
      (ensures
        MS.__matmul_single (mtranspose b) (mtranspose a) i j to ==
        MS.__matmul_single a b j i to)
      (decreases to)
  = if to = 0 then begin
      MS.matmul_zero_lemma (mtranspose b) (mtranspose a) i j;
      MS.matmul_zero_lemma a b j i
    end else begin
      matmul_transpose_partial a b i j (to - 1);
      MS.matmul_single_lemma (mtranspose b) (mtranspose a) i j to;
      MS.matmul_single_lemma a b j i to
    end
#pop-options

let matmul_transpose_cell
  (#n : nat)
  (a b : chest2 real n n)
  (i j : natlt n)
  : Lemma
      (ensures
        acc2 (MS.matmul (mtranspose b) (mtranspose a)) i j ==
        acc2 (mtranspose (MS.matmul a b)) i j)
  =
  matmul_transpose_partial a b i j n;
  MS.lemma_matmul_index (mtranspose b) (mtranspose a) i j;
  MS.lemma_matmul_index a b j i

let matmul_transpose
  (#n : nat)
  (a b : chest2 real n n)
  : Lemma
      (ensures
        MS.matmul (mtranspose b) (mtranspose a) ==
        mtranspose (MS.matmul a b))
  =
  Classical.forall_intro_2 (matmul_transpose_cell a b);
  assert (Kuiper.Chest.equal
    (MS.matmul (mtranspose b) (mtranspose a))
    (mtranspose (MS.matmul a b)))

inline_for_extraction noextract
fn lower_triangular_matmul
  (#et:Type0) {| floating et, real_like et |}
  (n : szp { n * n <= max_blocks * max_threads })
  (gA : array2 et (l2_row_major n n) { is_global gA })
  (gB : array2 et (l2_row_major n n) { is_global gB })
  (gC : array2 et (l2_row_major n n) { is_global gC })
  (#sA #sB #sC : chest2 et n n)
  (#rA #rB : chest2 real n n)
  preserves
    cpu ** on gpu_loc (gA |-> sA) ** on gpu_loc (gB |-> sB)
  requires
    on gpu_loc (gC |-> sC) **
    pure (
      sA %~ rA /\ sB %~ rB /\
      is_lower_triangular rA /\ is_lower_triangular rB)
  ensures
    exists* (sC' : chest2 et n n).
      on gpu_loc (gC |-> sC') **
      pure (sC' %~ MS.matmul rA rB)
{
  mtranspose_approx sA rA;
  mtranspose_approx sB rB;
  lower_transpose_is_upper rA;
  lower_transpose_is_upper rB;

  map_loc gpu_loc (fun () -> TGT.ghost_transpose1 gA);
  map_loc gpu_loc (fun () -> TGT.ghost_transpose1 gB);
  map_loc gpu_loc (fun () -> TGT.ghost_transpose1 gC);

  upper_triangular_matmul n
    (TGT.row2col gB) (TGT.row2col gA) (TGT.row2col gC)
    #(mtranspose sB) #(mtranspose sA) #(mtranspose sC)
    #(mtranspose rB) #(mtranspose rA);
  with sCT. assert on gpu_loc (TGT.row2col gC |-> sCT);

  mtranspose_involutive sA;
  mtranspose_involutive sB;
  map_loc gpu_loc (fun () -> TGT.ghost_transpose1_back gA);
  map_loc gpu_loc (fun () -> TGT.ghost_transpose1_back gB);
  map_loc gpu_loc (fun () -> TGT.ghost_transpose1_back gC);

  matmul_transpose rA rB;
  mtranspose_approx sCT
    (MS.matmul (mtranspose rB) (mtranspose rA));
}
