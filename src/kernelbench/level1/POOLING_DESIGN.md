# Pooling cluster — design notes (KernelBench L1 #41–#46)

**Status:** spec landed (`Kuiper.Spec.Pool1D` + `Kuiper.Spec.Pool2D`),
monoid-reduction infrastructure landed (`Kuiper.Monoid.Reduce` +
`Kuiper.Monoid.Reduce.F32`), GPU primitive `Kuiper.Kernel.WindowReduce1D`
**still deferred**.
**Owners (resume target):** any session picking up #41 or #44.

This document captures the design decisions made while triaging the
KernelBench L1 pooling cluster (problems 41–46) so the next session
can resume without re-doing the analysis.  The triage conclusion was
that a verified `WindowReduce1D` Pulse primitive is a multi-day
engagement.  Two rounds of preparatory work have landed:

  - `src/lib/spec/Kuiper.Spec.Pool1D.{fst,fsti}` — pure functional
    spec for max-pool / avg-pool with kernel / stride / padding /
    dilation, scalar-polymorphic, exact-fmax / approximate-sum.
  - `src/lib/spec/Kuiper.Spec.Pool2D.{fst,fsti}` — separable 2-D
    pool postcondition defined as the composition of two 1-D
    window reductions along H and W (the natural shape produced
    by a separable kernel implementation).  Per-axis predicates
    `row_{max,avg}_pooled_along_w` and `col_{max,avg}_pooled_along_h`.
  - `src/lib/kuiper/Kuiper.Monoid.Reduce.{fst,fsti}` and
    `Kuiper.Monoid.Reduce.F32.fsti` — fold-style commutative
    monoid abstraction with `cmonoid_fmax_f32` / `cmonoid_fadd_f32`
    interface-only instances.  This is the "reduction monoid"
    typeclass the `WindowReduce1D` primitive will consume.
    (Distinct from the existing append-style `Kuiper.Monoid.monoid0`,
    which is unsuited for fold-style accumulation.)
  - `src/lib/kuiper/Kuiper.Math.Fmax.fsti` — narrower `fmax`
    reduction infrastructure for `f32` (axiomatised assoc / comm /
    `neg_inf` neutrality, sequence reducer + three derived lemmas).
    Landed in parallel for the #49 max-reduction work; usable by
    pool #41 / #42 / #43 too.
  - `src/kernelbench/level1/challenge{41..46}/STATUS.txt` — short
    per-challenge pointers back to this doc.

All the above verify clean under `make ADMIT=0`.

No stub `.fst` impls have been committed.  No Python / bridge / run
scripts have been added for kernels that don't exist yet.

---

## 1. The six problems at a glance

The shapes / params are extracted directly from
`KernelBench/KernelBench/level1/4{1..6}*.py` (Apr 2026 snapshot).

|  # | Op           | Shape (B, C, *spatial*)        | K | S | P | D | Notes                              |
| -: | :----------- | :----------------------------- | :-: | :-: | :-: | :-: | :--------------------------------- |
| 41 | MaxPool1d    | (64, 192, 65536)               | 8 | 1 | 4 | 3 | dilation=3, large L                |
| 42 | MaxPool2d    | (32, 64, 512, 512)             | 4 | 1 | 1 | 1 | tiny K, square                     |
| 43 | MaxPool3d    | (16, 32, 128, 128, 128)        | 3 | 2 | 1 | 3 | dilation=3 in 3-D                  |
| 44 | AvgPool1d    | (64, 128, 65536)               | 8 | 1 | 4 | 1 | no dilation; large L               |
| 45 | AvgPool2d    | (16, 64, 2048, 2048)           |11 |11 | 0 | 1 | stride=K (default), no dilation    |
| 46 | AvgPool3d    | (16, 32, 128, 128, 256)        | 3 | 2 | 1 | 1 | no dilation; 3-D                   |

(`stride=K` for #45 is PyTorch's default-stride behaviour — the
constructor only specifies `kernel_size=11`.)

Output spatial sizes follow the standard PyTorch formula

```
L_out = floor((L + 2P - D*(K-1) - 1) / S) + 1
```

per spatial axis, computed in `Kuiper.Spec.Pool1D.pool_out_len_1d`.

**Important PyTorch quirks** the spec models:

- `MaxPool*d`: padded positions effectively contribute `-inf` (so the
  max never picks them).  Our spec encodes this by *skipping*
  out-of-bounds positions (the [option t] accumulator pattern in
  `window_fmax_aux`) — equivalent for finite inputs, avoids needing
  to introduce a sentinel scalar.
- `AvgPool*d`: default `count_include_pad=True`, so the divisor is
  always `K` (per axis) and padded positions contribute `0` to the
  numerator.  Our spec encodes this directly in `avg_window_sum_r`.

---

## 2. Primitive shape: `Kuiper.Kernel.WindowReduce1D`

The proposed shared primitive — *not yet implemented* — has this
abstract shape, designed to cover all six challenges by reduction
to a stack of 1-D row-wise window reductions:

```
val window_reduce_1d
  (#et : Type0) {| scalar et |}
  (#m : monoid et)               (* fmax for max-pool, fadd for avg-pool *)
  (k s p d : sz)                 (* kernel, stride, pad, dilation *)
  (input  : Array2.t et l_in)    (* shape (B*C, L)        *)
  (output : Array2.t et l_out)   (* shape (B*C, L_out)    *)
  (#sx  : chest2 et bc l)
  ...
  requires
    on gpu_loc (input  |-> Frac f sx)  **
    on gpu_loc (output |-> v)
  ensures
    on gpu_loc (input  |-> Frac f sx) **
    on gpu_loc (output |-> sx')        ** pure (windowreduce_post m k s p d sx sx')
```

The monoid typeclass is the existing `Kuiper.Monoid` (already in
tree at `src/lib/kuiper/Kuiper.Monoid.fst`).  Two instances would be
needed:

- `monoid_fmax  : monoid et` — identity = "no element" (encoded as
  the existing `option et` accumulator pattern, or as a
  caller-supplied sentinel scalar e.g. `min_value`); op = `fmax`.
- `monoid_fadd  : monoid et` — identity = `zero`; op = `add`.  The
  approximate-real spec falls out via the existing
  `Kuiper.Approximates` `a_add` lemma.

**2-D / 3-D pooling reduces to two / three calls** with appropriate
(B*C, H*W) → (B*C*W, H) reshaping (a logical reshape; the underlying
buffer is row-major and contiguous for the natural stride pattern).
For #45 (`stride=K=11, P=0`) and #46 (3-D, sep), this composition is
clean.  For non-separable cases (e.g. arbitrary 2-D max-pool with
non-trivial padding) the composition still holds because both
`fmax` and `fadd` are commutative+associative monoids and the 2-D
window factorises into independent 1-D windows along each axis.

The shape claim above (`output : Array2 of shape (B*C, L_out)` with
`L_out = pool_out_len_1d L K S P D`) is what the spec
`maxpool1d_post` / `avgpool1d_post` already enforce.

### 2.1 Setup / teardown sketch

Each output thread owns one output element; given small K (≤ 11
in this cluster), the thread reads K input elements directly with
no shared-memory cooperation.  The setup splits the input row's
read permission across `L_out` per-thread fractions and the output
row's write permission likewise.

For maxpool with K=8, S=1 (#41), each input element is read by up
to 8 threads → permission must be split 8-way (a `Frac` of 1/8 per
thread).  This is a frame-fact-only setup, well within reach of the
existing `forevery_factor'` machinery used by `RowScale` / GEMM
kernels.

For #45 (K=11, S=11), each input element is read exactly once → no
fractional-permission split needed at all; setup is trivial.

The teardown gathers the per-thread output fractions back into one
write permission on the output row, mirroring the existing
`gpu_matrix_gather_n` pattern.

Pseudocode for the per-thread kernel body:

```pulse
fn pool_thread_body
  (#et : Type0) {| scalar et |} {| floating et |}
  (input_row  : ...) (output_row : ...) (j : sz)
  (k s p d : sz)
  ...
{
  let mut acc : et = identity_of_monoid;
  for di in 0 .. k {
    let raw : int = j*s + di*d - p;
    if 0 <= raw && raw < l then
      acc := op_of_monoid acc (input_row[raw]);
    (* else: skip (max) or add 0 (avg) *)
  };
  output_row[j] := finalize_of_monoid acc; (* identity for max, mul-by-inv-K for avg *)
}
```

The key proof obligation per thread is that `acc` equals the
appropriate `window_fmax_aux` / `avg_window_sum_r` value at every
loop iteration — a straightforward inductive invariant over `di`.

### 2.2 Padding strategy

**Zero-pad on read via bounds-check predicate, no actual buffer
expansion.**  Allocating a padded shadow buffer would be a 2-pass
kernel (pad + reduce) and would double DRAM traffic for what is
already a memory-bound op.  The bounds check is per-K-iteration and
predictable (the same `0 <= raw < l` pattern for every output slot
in a row), so the branch is essentially free on modern GPUs.

For max-pool the out-of-bounds branch *skips* the update; for
avg-pool it would add 0 (i.e., is a no-op).  The spec encodes both
behaviours symmetrically (`pool_in_bounds` predicate threaded
through both `window_fmax_aux` and `avg_window_sum_r`).

### 2.3 Why max is exact and avg is approximate

- **Max-pool: exact.**  IEEE-754 `fmax` is commutative and
  associative on non-NaN inputs (and PyTorch pooling does not
  introduce NaNs from finite inputs).  Therefore *any* permutation
  of `fmax` over a fixed set of FP values yields a single
  bit-deterministic result, and the spec can pin the output by
  equality.  Our spec uses a left-fold (`window_fmax_aux`) but the
  result is order-independent.  This matches how `Kuiper.Spec.GEMM`
  pins integer-typed results bit-exactly while leaving FP results
  approximate — except here the operator itself is commutative, so
  even FP is exact.

- **Avg-pool: approximate (`%~`).**  IEEE-754 `+` is *not*
  associative, so different reduction orders (e.g., sequential
  per-thread vs. tree reduction across threads) produce different
  bit patterns.  We follow the existing pattern from
  `Kuiper.Spec.Frobenius` and `Kuiper.Spec.L2Norm`:
  existentially-bind the FP partial sum and require it to
  approximate the real-valued window sum via `s %~
  avg_window_sum_r row k s p d j`.  The output is then `mul s
  inv_k` for a caller-provided `inv_k` — same shape as
  `frobenius_result`'s uniform-scaling step.

---

## 3. LOC estimates

|  Component                            |  LOC | Status      |
| :------------------------------------- | -----: | :---------- |
| `Kuiper.Spec.Pool1D` (`.fst` + `.fsti`)  |  ~250 | ✅ landed   |
| `Kuiper.Spec.Pool2D` (`.fst` + `.fsti`)  |  ~250 | ✅ landed (compositional spec; rectangle equivalence deferred) |
| `Kuiper.Monoid.Reduce` (+ F32 instances) | ~140 | ✅ landed   |
| `Kuiper.Math.Fmax` (narrower fmax axioms) |  ~80 | ✅ landed   |
| `Kuiper.Kernel.WindowReduce1D` core   |  ~600 | ❌ deferred |
| Setup / teardown (split + gather)     |  ~250 | ❌ deferred |
| `Kuiper.Spec.Pool3D` separable spec    |  ~150 | ❌ deferred |
| Per-challenge `Kuiper.KB.MaxPool1D`   |  ~150 | ❌ deferred |
| Per-challenge `Kuiper.KB.AvgPool1D`   |  ~150 | ❌ deferred |
| 2-D / 3-D challenges (#42 #43 #45 #46) | 200 ea | ❌ deferred |
| Bridges + Python harnesses (×6)       |  ~120 ea | ❌ deferred |

The primitive is the long pole: ~1000 LOC of new Pulse kernel + proof
plus its monoid plumbing.  After it lands, each individual
KernelBench impl is ~150–200 LOC of mechanical wiring.

---

## 4. Why this was triaged out of the current session

The prior agent that started this cluster correctly observed that
the existing kernel primitives (`HReduce`, `HReduce.Block`,
`reduce_batched`, `reduce_batched_block`) all assume the reduction
axis is contiguous and full-row.  Pooling needs:

1. *windowed* reduction (K consecutive elements out of L), repeated
   `L_out` times per row, with overlapping windows when `S < K`;
2. *strided* output indexing;
3. *dilated* inner index computation;
4. *padding* via in-bounds predicate.

None of these can be reduced cleanly to the existing primitives.
A new dedicated primitive is required, and that primitive's
setup/teardown proof is non-trivial because the per-thread input
read permissions overlap (`S < K` — every input element can be
read by multiple output threads).  The `forevery_factor'` /
`forevery_zip` plumbing for an overlap of K-element-wide windows
with stride-S between them is the bulk of the engagement.

Verifying the kernel itself (the inner loop with the bounds-check
predicate) is comparatively easy once the permission split is in
place.

---

## 5. Resume instructions

When picking this back up:

1. **Re-read this doc and the spec** — `Kuiper.Spec.Pool1D` is
   already polymorphic over the input scalar type and is the
   intended postcondition for `windowreduce_post`.

2. **Decide whether to start with #44 or #45.**  #45 has the
   simplest permission pattern (`S = K`, no overlap, no dilation,
   no padding → trivial setup).  #44 has overlap (`S = 1, K = 8`)
   and padding but no dilation.  #45 first → use it as a template
   for the primitive's "easy path", then generalise to #44.

3. **Land `monoid_fadd` and `monoid_fmax` first** as standalone
   typeclass instances — they're independently useful.
   ✅ **DONE** as `Kuiper.Monoid.Reduce.F32.cmonoid_fmax_f32` and
   `cmonoid_fadd_f32`, plus a narrower `Kuiper.Math.Fmax` for the
   `seq_fmax` reducer used by tree reductions.

4. **Implement `Kuiper.Kernel.WindowReduce1D`** following the
   sketch in §2.1 above.  The setup is the hard part; copy
   structure from `Kuiper.Kernel.RowScale.setup` and adjust the
   `forevery_factor'` arity to K-fold (or 1-fold for `S = K`).

5. **Wire challenge #45 first** as the sanity check (no overlap,
   no padding).  Once it verifies + extracts + runs, generalise
   to the other five.  The 2-D / 3-D variants compose 1-D calls;
   they don't need a new primitive.

6. The avg-pool challenges need an `inv_k : t` host scalar
   computed as `div one (count_t k)` where `count_t` casts the
   nat `K` to `et`.  This is a standard pattern (see
   `Kuiper.KB.LayerNorm` for the analogue with `inv_n`).

7. **Don't try to ship all six in one session** — even with the
   primitive landed, that's a week of work.  Treat each KB
   challenge as a separate ~200-LOC PR.

---

## 6. Spec module — quick reference

`Kuiper.Spec.Pool1D` exposes:

- `pool_out_len_1d (l k s p d : nat) : nat` — output length.
- `pool_in_bounds  (l s p d j di : nat) : bool` — per-tap bounds.
- `pool_input_idx  (l s p d j di : nat) : Pure nat` — refined index.
- `avg_window_sum_r` — real-valued window sum (out-of-bounds = 0).
- `window_fmax_aux` / `max_window` — `fmax` fold over in-bounds taps.
- `row_max_pooled_1d` / `row_avg_pooled_1d` — per-row predicates.
- `maxpool1d_post`   / `avgpool1d_post`   — whole-tensor postconditions.

All defined in terms of `Kuiper.Scalars.scalar`, `Kuiper.Scalars.floating`
and `Kuiper.Approximates.real_like` typeclasses; nothing pool-specific
escapes into the wider type vocabulary.

---

## 7. Why 2-D/3-D pool still does N passes "in the driver" — and what verifying it away costs

**Question raised (review):** "Why is #42 doing two passes in the driver?
Verify as much as possible."

**Answer.** The only verified pooling primitive is
`Kuiper.Kernel.WindowReduce1D`, which reduces over the *inner (last)*
axis of a 2-index `Array2.layout (rows, l)` view. 2-D max-pool is realised
by separable composition (§2):

  - pass 1 reduces over W — already the inner axis → verified directly;
  - pass 2 must reduce over H, which is *not* inner, so the intermediate
    `(bc, H, W_out)` buffer is transposed to `(bc, W_out, H)` (H now inner),
    reduced, then transposed back.

Those **two transposes are PyTorch `.permute().contiguous()` calls in the
C++ bridge** — they are the reason the orchestration lives "in the driver"
rather than inside a single verified entry. `WindowReduce1D` preserves the
row count between input and output (it only windows the inner axis), so it
*cannot itself transpose*; and Kuiper currently has **no executable
transpose kernel** (only the ghost `Kuiper.Ghost.{Transpose,TensorTranspose}`).

### What it would take to fold the orchestration inside the verified boundary

Three independent, each **non-trivial / multi-day** options (none is a quick
edit — all need new verified-kernel or layout-proof infrastructure, so they
are flagged here for maintainer scheduling rather than done unilaterally):

1. **Custom flattened-batched-col-major 2-index layout.** Give pass 2 an
   input layout `(bc*W_out, H)` whose `imap(R,h) = (R/W_out)*(H*W_out) +
   h*W_out + (R%W_out)` reads the `(bc,H,W_out)` buffer strided — no physical
   transpose. This map is a valid injection but is **not** a nested-major
   (`major_on`) layout, so it cannot be built from `Tensor.Layout.Alg`; it
   needs a hand-written `layout_f_for` injection + injectivity proof + a
   concrete `auto_cinj`/`ctlayout` instance, plus re-deriving
   `windowreduce_result` over it. Eliminates *both* permutes; smallest of the
   three if the bespoke-layout proof goes through. (`l2_col_major` /
   `Array2.Strided.strided_col_major` only handle a *single* plane, bc=1;
   `subtile_layout` would force one launch per plane — up to ~10^6 — so
   neither suffices off the shelf.)

2. **Verified transpose-via-strided-copy kernel.** A new out-of-place copy
   kernel (input arbitrary `ctlayout`, output row-major) = a verified
   transpose; then a single verified `fn maxpool2d_full` does
   pass1→transpose→pass2→transpose-back internally. Removes the unverified
   PyTorch permutes but keeps the data shuffle, and still needs the
   composition lemma (below) for an end-to-end 2-D post.

3. **True 2-D rectangle `WindowReduce2D` kernel.** One launch computes the
   `k_h × k_w` max directly from `(bc,H,W)` row-major → `(bc,H_out,W_out)`,
   no transpose, no passes. Cleanest result (driver does nothing) but the
   largest proof effort — comparable to the original multi-day `WindowReduce1D`
   bring-up (§3).

### The other half of "verify as much as possible": the composition lemma

Even options 1–2 only give "two verified 1-D passes + verified data movement".
The claim *"max over W then max over H = the 2-D rectangle max"* is the
`maxpool2d_post` ⇔ rectangle equivalence, which `Kuiper.Spec.Pool2D` defines
**compositionally** and whose rectangle-equivalence lemma is explicitly
**deferred** (it reduces to `fmax` assoc/comm, which `Pool1D` does not yet
expose; see the `Kuiper.Spec.Pool2D.Rect` note in that module's header). A
fully-verified end-to-end 2-D post needs that lemma regardless of which
transpose-elimination option is chosen.

**Recommendation:** option 1 (bespoke batched-col-major layout) as the best
effort/reward — it removes the unverified permutes with no data shuffle and no
new kernel — paired with landing the `Pool2D.Rect` `fmax` assoc/comm lemma to
discharge `maxpool2d_post`. Both are sizeable enough to deserve their own
session. The same analysis applies verbatim to #43 (3 passes, 2 permutes) and
to AvgPool #45/#46 (with `fadd` in place of `fmax`).

### Update — option 1 LANDED for #42 (transpose-free single entry)

Implemented and committed (`e91c4916`, building on milestone-1 `2b421a30`):

- **Single verified entry** `maxpool2d_full_alloc_f32` in
  `challenge42/Kuiper.KB.MaxPool2D.fst`. It internally runs pass 1
  (`windowreduce` over W, `l2_row_major`), allocates the row-major
  `(B*C,H_out,W_out)` output, runs pass 2 reading the row-major
  `(B*C,H,W_out)` intermediate through the flat batched-col-major view and
  writing strided into the output, frees the intermediate, and returns
  `(W_out, (H_out, out_ptr))`. **No physical transpose.**
- The flat batched-col-major view is the **existing abstract
  `Kuiper.Tensor.Layout.BCMPages.l2_bcm_pages b hw c`** (dims `(b*hw, c)`,
  imap `(r/hw)*(c*hw)+ci*hw+(r%hw)`), used with `b=B*C`, `hw=W_out`,
  `c=H` (input view) / `c=H_out` (output view). The hand-rolled
  `Kuiper.Array2.BatchedColMajor.flat_bcm` (milestone 1) has the identical
  imap and verifies standalone, **but does not extract**: its standalone
  refined `cimap` helper produces a polymorphic tuple coercion KRML cannot
  monomorphize (Warning 26 `<: any`), which globally poisons extraction of
  even unrelated clean kernels. Lesson: for anything that must extract, prefer
  the abstract `[@@erasable] val` BCMPages/BCMChannels layouts over
  `pack (mk_injection ...)` hand-rolls.
- The bridge (`kuiper_maxpool2d_bridge.cu`) now makes **one** call — the
  unverified PyTorch `.permute().contiguous()` inter-pass transposes are gone.
  The generated pass-2 kernel reads `mid2[r/wo*h*wo + dpos*wo + r%wo]` and
  writes `out[r/wo*ho*wo + j*wo + r%wo]` — the strided-H read/write with no
  shuffle, exactly as intended. Verifies at z3rlimit<=40; **#42 test PASSes**
  (fp32 atol=rtol=1e-4, RTX A6000).

**Remaining gap (unchanged):** the post is the two per-axis
`windowreduce_result` predicates, NOT the end-to-end `maxpool2d_post`. The
latter still needs the deferred `Pool2D.Rect` `fmax` assoc/comm lemma (above).
That, and the analogous #43/#45/#46, remain follow-ups.
