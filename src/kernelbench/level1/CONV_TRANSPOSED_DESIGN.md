# Transposed-convolution cluster — design notes (KernelBench L1)

**Status:** Phase 0 specs landed (`Kuiper.Spec.ConvTranspose1D`,
`Kuiper.Spec.ConvTranspose2D`, `Kuiper.Spec.ConvTranspose3D`).
GPU kernel impls **deferred**.

**Scope:** 17 challenges — `#57, #58, #61, #64, #65, #68, #69, #70,
#71, #72, #73, #74, #75, #77, #78, #79, #81`.

This document captures the design decisions made while triaging the
KernelBench L1 transposed-convolution cluster so the next session
can resume without re-doing the analysis.  Kernel impls are
multi-day each (the proof structure is comparable to `Conv2D`,
which has not yet been implemented either) — Phase 0 lands the
shared functional postconditions and the per-challenge dim table.

What landed (verified, ADMIT=0):

  - `src/lib/spec/Kuiper.Spec.ConvTranspose1D.{fst,fsti}`
  - `src/lib/spec/Kuiper.Spec.ConvTranspose2D.{fst,fsti}`
  - `src/lib/spec/Kuiper.Spec.ConvTranspose3D.{fst,fsti}`

  All three are scalar-polymorphic, mirror `Kuiper.Spec.Conv2D`,
  and compose with the existing `Kuiper.EMatrix` /
  `Kuiper.Approximates` machinery for FP-approximate post.

  - `src/kernelbench/level1/challenge{57,58,61,64,65,68-75,77-79,81}/STATUS.txt`
    — short per-challenge pointer back to this doc.

No stub `.fst` impls have been committed.  No Python harness or
bridge has been added for kernels that don't exist yet.

---

## 1. Form (a) vs form (b) — choice

Two equivalent definitions of transposed convolution are in common
use:

- **(a) Direct scatter / per-output-pixel sum.**  Each output
  element is a sum of products over `(ic, k...)`, reading from a
  *strided + zero-padded* view of the input.  No fictitious
  intermediate tensor.

- **(b) Standard convolution on stretched-zero / padded input.**
  Insert `(stride - 1)` zeros between input pixels, pad, then run
  a regular `Conv*d` with stride 1 and an adjusted padding.

We picked **form (a)** for the spec.

Rationale:

1. **Simpler types.**  Form (b) requires reasoning about a
   logical "stretched" tensor whose dimensions are
   `(L_in - 1)*S + 1 + 2*P_eff`.  Either we materialise that
   tensor (memory blowup at the spec level — ugly) or we model
   it via a refinement type and a derived read function — which
   is essentially form (a) with extra indirection.

2. **No dependency on `Conv2D` reuse.**  Form (b) would force the
   ConvTranspose2D spec to depend on `Conv2D`.  Form (a) only
   needs the existing tensor-extensionality scaffolding, which
   we get by sharing `etensor4` from `Kuiper.Spec.Conv2D` (and
   by defining sister `etensor3` / `etensor5` types in the new
   1-D / 3-D modules).

3. **Output_padding is trivial in form (a).**  See §4.

4. **Mirrors `Kuiper.Spec.Conv2D` structurally.**  The recursive
   `__convT*d_single` accumulator over a linearised
   `(ic, k...)` index is line-for-line analogous to
   `Kuiper.Spec.Conv2D.__conv2d_single`.  Reusing
   `unrank_ic / unrank_kh / unrank_kw` from `Kuiper.Spec.Conv2D`
   in the 2-D module saves boilerplate.

Form-(a) per-output formula (2-D case shown — 1-D and 3-D are
straightforward generalisations):

```
y[b, oc, oh, ow] = bias[oc]
                 + Σ_{ic, kh, kw}  x'[b, ic, oh + P_h - kh*D_h,
                                              ow + P_w - kw*D_w]
                                   * w[ic, oc, kh, kw]

x'[b, ic, h_num, w_num] =
   x[b, ic, h_num/S_h, w_num/S_w]
     if h_num >= 0 ∧ w_num >= 0
        ∧ h_num % S_h = 0 ∧ w_num % S_w = 0
        ∧ h_num/S_h < H_in ∧ w_num/S_w < W_in
   zero otherwise
```

The "`h_num` divisible by `S_h`" condition is what makes
form (a) equivalent to "stretch the input by `S` then convolve":
indices that don't fall on the stretched lattice see zeros.

The spec is exact (scalar-typed, `add`/`mul`/`zero` from the
`scalar` typeclass).  Lift to the FP-approximate post via the
existing `Kuiper.Approximates.real_like` lemmas, identically to
how `Kuiper.Spec.GEMM` and `Kuiper.Spec.Conv2D` do it.

### 1.1 Weight layout — note the transpose

For `nn.ConvTranspose*d`, PyTorch stores the weight as

  `weight : (C_in, C_out_per_group, K...)`

(transposed in the channel axes compared to `nn.Conv*d`, which
uses `(C_out, C_in_per_group, K...)`).  Our spec reflects that:
`weight : etensorN et cin cout k...`.  Impls must match this
layout when interfacing with PyTorch checkpoints, but no axis
permutation is needed inside the kernel — the spec is the
ground truth.

---

## 2. Per-challenge dim table

All shapes and parameters extracted directly from
`KernelBench/KernelBench/level1/{nn}_*.py` snapshots.

Notation: dims listed as `(B, C_in, C_out)` for batch / channels
followed by spatial.  `K`, `S`, `P`, `D`, `OPad` follow PyTorch
convention.  `G` = groups.  Bias is `False` everywhere in this
cluster.

|  # | Op  | (B, C_in, C_out) | Spatial in    | K        | S       | P       | D       | OPad    | G | Special           |
| -: | :-: | :---------------: | :------------ | :------- | :------ | :------ | :------ | :------ | -: | :---------------- |
| 57 | T2D | (8, 64, 64)       | 1024×1024     | (3, 3)   | (1, 1)  | (0, 0)  | (1, 1)  | (0, 0)  | 1 | square, default   |
| 58 | T3D | (16, 32, 16)      | 16×32×64      | (3, 5, 7)| (1,1,1) | (0,0,0) | (1,1,1) | (0,0,0) | 1 | asym kernel       |
| 61 | T3D | (8, 48, 48)       | 64×64×64      | (3,3,3)  | (1,1,1) | (0,0,0) | (1,1,1) | (0,0,0) | 1 | square            |
| 64 | T1D | (64, 128, 128)    | 65536         | 3        | 1       | 0       | 1       | 0       | 1 | default 1-D       |
| 65 | T2D | (8, 64, 64)       | 512×512       | (3, 7)   | (1, 1)  | (0, 0)  | (1, 1)  | (0, 0)  | 1 | asym kernel       |
| 68 | T3D | (16, 32, 64)      | 64×64×64      | (3, 5, 5)| (1,1,1) | (0,0,0) | (1,1,1) | (0,0,0) | 1 | asym kernel       |
| 69 | T2D | (64, 64, 128)     | 128×256       | (3, 5)   | (1, 1)  | (0, 0)  | (1, 1)  | (0, 0)  | 1 | asym in/out       |
| 70 | T3D | (8, 48, 24)       | 96×96×96      | (3,3,3)  | (1,1,1) | (0,0,0) | (1,1,1) | (0,0,0) | 1 | asym B/C          |
| 71 | T2D | (8, 32, 32)       | 512×1024      | (3, 3)   | (1, 1)  | (0, 0)  | (1, 1)  | (0, 0)  | 1 | asym in           |
| 72 | T3D | (8, 32, 32)       | 12×24×48      | (3, 5, 7)| (2,2,2) | (1,2,3) | (1,1,1) | (1,1,1) | 4 | grouped+strided+OP|
| 73 | T3D | (4, 32, 32)       | 32×64×128     | (3,3,3)  | (2,2,2) | (1,1,1) | (1,1,1) | (0,0,0) | 4 | grouped+strided   |
| 74 | T1D | (32, 32, 64)      | 131072        | 5        | 1       | 0       | 3       | 0       | 1 | dilated 1-D       |
| 75 | T2D | (16, 32, 64)      | 128×256       | (3, 5)   | (2, 3)  | (1, 2)  | (2, 1)  | (0, 0)  | 4 | grouped+everything|
| 77 | T3D | (16, 32, 64)      | 16×32×32      | (3,3,3)  | (2,2,2) | (1,1,1) | (2,2,2) | (0,0,0) | 1 | strided+dilated   |
| 78 | T2D | (8, 32, 32)       | 512×1024      | (3, 7)   | (1, 1)  | (1, 3)  | (1, 1)  | (0, 0)  | 1 | padded            |
| 79 | T1D | (16, 32, 64)      | 131072        | 3        | 2       | 1       | 2       | 0       | 1 | strided+pad+dil   |
| 81 | T2D | (16, 32, 64)      | 64×128        | (3, 3)   | (5, 5)  | (1, 1)  | (2, 2)  | (0, 0)  | 1 | strided+dilated   |

For every challenge the output shape is

```
L_out = (L_in - 1) * S - 2*P + D*(K - 1) + OPad + 1   (per axis)
```

computed via the helpers
`Kuiper.Spec.ConvTranspose1D.convT1d_out_len` (1-D) and
`Kuiper.Spec.ConvTranspose2D.convT_out_len_1d` (per-axis;
re-used by the 3-D module's caller for each of D/H/W).

### 2.1 Cluster splits

By dimensionality:

  - **1-D (3 challenges):** #64, #74, #79
  - **2-D (7 challenges):** #57, #65, #69, #71, #75, #78, #81
  - **3-D (7 challenges):** #58, #61, #68, #70, #72, #73, #77

By feature combination:

  - **Vanilla (S=1, P=0, D=1, G=1):** #57, #58, #61, #64, #65,
    #68, #69, #70, #71 — 9 challenges.  Same kernel, parametrised
    by dims.
  - **Padded only (S=1, D=1, G=1):** #78
  - **Strided + dilated (G=1):** #74, #77, #79, #81
  - **Grouped + strided + (dilated/padded):** #72, #73, #75 —
    these need the extra channel-split logic; see §3.

### 2.2 Reuse pattern across vanilla challenges

For the 9 vanilla `(S, P, D, G) = (1, 0, 1, 1)` challenges, the
spec body specialises to

```
y[b, oc, oh, ow] = bias[oc]
                 + Σ_{ic, kh, kw} x_pad[b, ic, oh-kh, ow-kw]
                                   * w[ic, oc, kh, kw]
```

i.e. simple zero-padded reads, no divisibility check needed.  A
single Pulse kernel parametrised by `(B, C_in, C_out, H_in, W_in,
K_h, K_w)` covers all of them — only the host-side bridge differs.

Even non-vanilla challenges can share most of the kernel structure
because the per-tap inner loop is the same; only the index
arithmetic at the read site changes.  This matters for LOC
estimates: total kernel work is roughly **3 × (one shared kernel
core)**, not **17 × (one kernel each)**.

---

## 3. Grouped variants — channel-split strategy

For challenges with `groups > 1` (#72 G=4, #73 G=4, #75 G=4) the
PyTorch contract is:

  - Input  partitions into `G` channel-groups of size `C_in / G`.
  - Output partitions into `G` channel-groups of size `C_out / G`.
  - The weight for `ConvTranspose*d` has shape
    `(C_in, C_out / G, K...)` — the second axis is the *per-group*
    output channel count.
  - Convolution within each group is independent of the others.

The group `g` for an output channel `oc` is `oc / (C_out/G)`; the
group's input-channel range is `[g*(C_in/G), (g+1)*(C_in/G))`.

**Spec strategy (NOT yet implemented):** add a thin wrapper
`convT2d_grouped : ... -> groups:pos -> ...` that, for each
output channel `oc`, restricts the `ic`-fold to the group's
input-channel range and indexes weight accordingly.  The wrapper
is a ~30-LOC addition per dimensionality (1-D, 2-D, 3-D) and only
needs to land before the grouped impls do.

**Impl strategy:** the grouped variants are the same kernel as
the ungrouped one, with the per-thread `ic` loop bounded to the
thread's group's input-channel range.  No new permission plumbing
is needed — the input is split by groups along the channel axis,
and the existing `forevery_factor'` machinery handles per-group
read shares cleanly (this is how grouped Conv2D would land too,
once it does).

If the impl agent prefers, an alternative is to materialise the
grouped problem as `G` independent ungrouped `ConvTranspose*d`
calls on the per-group sub-tensors, then concatenate.  This is
simpler at the host level but doubles the kernel-launch count —
fine for correctness, suboptimal for performance.  Either choice
is valid and the spec wrapper is independent of which is picked.

---

## 4. Output_padding parameter

`output_padding` is a PyTorch quirk that adds extra rows/columns
on the *bottom-right* of each output spatial axis, used to
disambiguate the "which output size does this stride-S
ConvTranspose produce" question.

In the form-(a) spec it requires **zero** machinery.  The output
spatial size formula

```
L_out = (L_in - 1)*S - 2*P + D*(K - 1) + OPad + 1
```

embeds `OPad` directly.  For output positions in the extra
`OPad` band, the per-output sum naturally contains only zeros
because every term's numerator either fails the divisibility
check (when `S > 1`) or falls outside `[0, L_in*S)`.  No spec
case-split, no extra parameter inside the recursion.

Concretely, for #72 (`OPad = (1, 1, 1)`) the output tensor has
one extra trailing slice along each spatial axis filled with
`bias[oc]` (since the inner sum is zero there), exactly matching
PyTorch.

The impl just needs to allocate the larger output buffer and run
the same kernel; the bias-only trailing band falls out of the
kernel's per-output-cell loop with no change.

---

## 5. LOC estimates

|  Component                                   |  LOC   | Status        |
| :------------------------------------------- | -----: | :------------ |
| `Kuiper.Spec.ConvTranspose1D` (`.fst`+`.fsti`) |  ~250 | ✅ landed     |
| `Kuiper.Spec.ConvTranspose2D` (`.fst`+`.fsti`) |  ~300 | ✅ landed     |
| `Kuiper.Spec.ConvTranspose3D` (`.fst`+`.fsti`) |  ~390 | ✅ landed     |
| Grouped-variant spec wrappers (3 dims × ~50)  |  ~150 | ❌ deferred   |
| Shared 2-D core kernel                        |  ~700 | ❌ deferred   |
| Shared 1-D core kernel                        |  ~500 | ❌ deferred   |
| Shared 3-D core kernel                        | ~1000 | ❌ deferred   |
| Setup / teardown (per dim, per stride class)  | ~250 ea | ❌ deferred |
| Per-challenge wiring (`Kuiper.KB.ConvT*`)     | ~150 ea | ❌ deferred |
| Bridges + Python harnesses (×17)              | ~120 ea | ❌ deferred |

The 2-D core is the long pole — once it lands, the 1-D variant is
a degenerate case (drop one axis) and the 3-D variant is the same
recursion with one extra axis.  A reasonable plan is therefore to
do all three core kernels back-to-back rather than 17 one-off
challenge impls.

---

## 6. Reuse opportunities w/ the (still-unwritten) `Conv2D.Naive`

`Kuiper.Spec.Conv2D` has been landed but no GPU kernel has shipped
yet.  When `Kuiper.Kernel.Conv2D.Naive` does land, the
ConvTranspose2D core can re-use:

  - `read_padded` analogue (we have `read_strided_padded_2d`,
    which degenerates to `read_padded` at `(S_h, S_w) = (1, 1)`
    with no divisibility check).
  - The `(ic, kh, kw)` linearised loop body (identical mul-add).
  - The per-thread output partition + shared bias broadcast.
  - The setup/teardown that splits input read permission across
    the `KH * KW` reading threads per input pixel — except that
    for ConvTranspose every input pixel is read by *exactly*
    `KH * KW` output positions (no boundary truncation because
    the strided + padded read function returns zero for OOB taps,
    so the same permission split holds for *every* input pixel).
    This is actually a **cleaner** permission story than for
    forward Conv2D, where boundary input pixels contribute to
    fewer output pixels.

Concretely: when `Conv2D.Naive` is implemented, the
ConvTranspose2D `(S, P, D, G) = (1, 0, 1, 1)` kernel for #57,
#65, #69, #71 can be done by **swapping the weight axis order
and inverting the index sign** in the existing `Conv2D.Naive`
kernel body.  That's the single biggest reuse opportunity.

---

## 7. Resume order

Suggested order when impl work resumes:

1. **#57** — simplest 2-D vanilla case (square 3×3, S=1, P=0).
   Lands the shared 2-D kernel.  Useful sanity check that the
   form-(a) spec composes with the existing `etensor4` /
   `forevery_factor'` machinery.

2. **#64 → #74 → #79** — 1-D family in increasing complexity
   (#64 vanilla, #74 adds dilation, #79 adds stride+pad+dil).
   The 1-D kernel is the easiest to reason about (no nested
   axes) and is a good place to land the divisibility-check
   read function before tackling 2-D non-vanilla cases.

3. **#65, #69, #71** — 2-D vanilla with asymmetric kernels /
   inputs.  Re-uses #57's kernel verbatim.

4. **#78, #81** — 2-D padded / strided+dilated.  Adds the
   stride/dilation index path to the 2-D kernel.

5. **#75** — 2-D grouped+strided+padded+dilated.  Lands the
   grouped wrapper.  Should fall out cleanly once #78/#81 verify.

6. **#61, #58, #68, #70** — 3-D vanilla.  Lands the 3-D kernel
   core.

7. **#77** — 3-D strided+dilated.

8. **#73, #72** — 3-D grouped variants.  #72 is the absolute
   worst case (every feature on, asymmetric, output_padding).

The vanilla 2-D and 3-D challenges should be batched into single
kernel implementations parametrised by dim, not implemented one
at a time.

---

## 8. Spec module — quick reference

`Kuiper.Spec.ConvTranspose1D` exposes:

  - `etensor3` / `mkT3` / `tacc3` — 3-D erased tensor scaffolding.
  - `convT1d_out_len  l_in s p d k opad : nat` — output length.
  - `read_strided_padded_1d` — strided + zero-padded read.
  - `unrank1d_ic` / `unrank1d_k` — `(ic, k)` index decomposition.
  - `__convT1d_single` — primitive recursive accumulator.
  - `__convT1d_single_zero_lemma` / `_lemma` — base/step lemmas.
  - `convT1d_single` — per-pixel result with bias.
  - `convT1d` — full output tensor.
  - `lemma_convT1d_index` — pointwise post.

`Kuiper.Spec.ConvTranspose2D` and `Kuiper.Spec.ConvTranspose3D`
are structurally identical with axis count adjusted (`etensor4`
reused from `Kuiper.Spec.Conv2D`; `etensor5` defined locally).

All defined in terms of `Kuiper.Scalars.scalar`; nothing
ConvTranspose-specific escapes into the wider type vocabulary.
The exact-equality post composes with
`Kuiper.Approximates.real_like` to recover an FP-approximate
post for fold-order-tolerant impls, identical to how
`Kuiper.Spec.GEMM` and `Kuiper.Spec.Conv2D` do it.
