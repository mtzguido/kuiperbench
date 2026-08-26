# Depthwise / pointwise / separable conv cluster — design notes (KernelBench L1 #82–#87)

**Status:** Phase 0 specs landed for all three primitives:

  - `Kuiper.Spec.DepthwiseConv2D.{fst,fsti}` — covers #82–#85 and the
    depthwise stage of #86.
  - `Kuiper.Spec.PointwiseConv2D.{fst,fsti}` — covers #87 and the
    pointwise stage of #86.
  - `Kuiper.Spec.SeparableConv2D.fst` — pure-F* compositional spec
    for #86, no new primitives.

All three verify `ADMIT=0` and contain no `assume` / `admit` / `magic`.

**No kernel implementations have been committed in this cluster.**
The Phase-1 wiring strategies are described below; #87 is the
unblocked entry point, the four depthwise challenges (#82–#85) wait
on a base `Conv2D` kernel landing.

---

## 1. The six problems at a glance

Shapes / params extracted directly from
`KernelBench/KernelBench/level1/8{2..7}*.py` (Apr 2026 snapshot).

|  # | Op                        | Input  (B, C_in, H, W)         | Out C / kernel  | S       | P       | D       | Bias  |
| -: | :------------------------ | :----------------------------- | :-------------- | :-----: | :-----: | :-----: | :---: |
| 82 | Depthwise 2D, sq/sq       | (16, 64, 512, 512)             | C_out = 64; K=3×3   | 1×1     | 0×0     | 1×1     | False |
| 83 | Depthwise 2D, sq/asym K   | (64, 8, 512, 512)              | C_out = 8 ; K=3×1   | 1×1     | 0×0     | 1×1     | False |
| 84 | Depthwise 2D, asym in/sq K| (64, 128, 256, 512)            | C_out = 128; K=3×3  | 1×1     | 0×0     | 1×1     | False |
| 85 | Depthwise 2D, asym/asym   | (32, 128, 128, 256)            | C_out = 128; K=3×7  | 1×1     | 0×0     | 1×1     | False |
| 86 | Depthwise-separable 2D    | (16, 64, 512, 512)             | C_out=128; dw K=3×3; pw 1×1 | 1×1     | 1×1 (dw)/0 (pw) | 1×1     | False |
| 87 | Pointwise 2D (1×1 conv)   | (16, 64, 1024, 1024)           | C_out = 128; K=1×1  | 1×1     | 0×0     | 1×1     | False |

All four depthwise challenges have `groups = in_channels = out_channels`,
i.e. the channel-multiplier is 1 — the spec is parameterised on this
case (`weight : etensor4 et C 1 K_h K_w`).

Output spatial dimensions follow the standard PyTorch formula
`H_out = floor((H + 2*P_h - D_h*(K_h-1) - 1) / S_h) + 1` per axis.
For #82–#87 with the stated params:

| #  | H_out | W_out | Notes                            |
|---:|:------|:------|:---------------------------------|
| 82 | 510   | 510   | K=3, P=0, S=1                    |
| 83 | 510   | 512   | K=(3,1), P=0, S=1                |
| 84 | 254   | 510   | K=3, P=0, S=1                    |
| 85 | 126   | 250   | K=(3,7), P=0, S=1                |
| 86 | 512   | 512   | K=3, P=1, S=1 (dw); pw preserves |
| 87 | 1024  | 1024  | 1×1 conv preserves spatial       |

---

## 2. Specs in tree

### 2.1 `Kuiper.Spec.DepthwiseConv2D`

Per-pixel formula:

```
y[b, c, oh, ow] = bias[c]
                + Σ_{kh, kw}
                     x[b, c,
                       oh*S_h + kh*D_h - P_h,
                       ow*S_w + kw*D_w - P_w]
                     * w[c, kh, kw]
```

Encodes channel-multiplier 1 (`weight : etensor4 et C 1 K_h K_w` —
the second axis is a placeholder so we can reuse the `etensor4`
algebra of `Spec.Conv2D`).  Reduction is left-fold over a linearised
`(kh, kw)` index, mirroring `Spec.Conv2D.__conv2d_single` minus the
`ic` axis.

Exports:

  - `__dwconv2d_single` (recursive partial sum, with zero-base + step lemmas)
  - `dwconv2d_single` (adds bias)
  - `dwconv2d` (whole-tensor `etensor4 et B C H_out W_out`)
  - `lemma_dwconv2d_index` (SMTPat: `tacc (dwconv2d ...) ...
                                 == dwconv2d_single ...`)

Bit-exact at `scalar et`; floating-point `%~` falls out of the
existing `real_like` / `a_add` / `a_mul` chain (same pattern as
`Spec.Conv2D`).

### 2.2 `Kuiper.Spec.PointwiseConv2D`

Per-pixel formula:

```
y[b, oc, h, w] = bias[oc]
               + Σ_{ic} w[oc, ic] * x[b, ic, h, w]
```

Reduction is over `C_in` only.  Weight is an `chest2 et C_out C_in`
(no spatial axes) — the natural shape for the existing GEMM kernels
to consume.  `__pwconv2d_single` is a left-fold over `ic`, mirroring
`Spec.GEMM.__matmul_single` exactly.

Exports analogous to `DepthwiseConv2D`:
`__pwconv2d_single` (+ zero-base + step lemmas), `pwconv2d_single`,
`pwconv2d`, `lemma_pwconv2d_index`.

### 2.3 `Kuiper.Spec.SeparableConv2D`

Pure F* composition, no `.fsti`:

```
separable_conv2d ... x dw_w dw_b pw_w pw_b
  = pwconv2d (dwconv2d ... x dw_w dw_b) pw_w pw_b
```

Plus `lemma_separable_conv2d_index` (one-liner).

---

## 3. Phase-1 wiring strategy per challenge

### 3.1 #87 (pointwise) — REUSES EXISTING GEMM KERNEL

Pointwise = per-pixel matmul.  The existing
`Kuiper.Kernel.GEMM.BlockTiling2D` (`g_gemm_f32_128x128x32_8x8`) is
exactly the right kernel: one matmul per batch produces an
`(C_out, H*W)` output slice.

NCHW layout makes the reshape free of permutation:

  - `x[b, :, :, :]` is contiguous in memory at stride `C_in * H * W`
    from the start of `x`, with shape `(C_in, H, W)` row-major.
  - View as `(C_in, H*W)` matrix — same memory, different shape, no
    copy needed.
  - Weight is `(C_out, C_in)` already — fits the GEMM kernel signature
    `(rows = C_out, shared = C_in, cols = H*W)`.

For #87 the matmul shape per batch is `(C_out=128, C_in=64) ×
(C_in=64, H*W=1048576)` — well-padded for the 128×128×32 BlockTiling2D
tile (rows=128 ≥ 128, cols ≥ 128, shared=64 ≥ 32).

**Bridge structure** (mirrors `challenge1/kuiper_gemm_bridge.cu`):

```cuda
torch::Tensor kuiper_pwconv_cuda(torch::Tensor x, torch::Tensor w) {
    int64_t B = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    int64_t C_out = w.size(0);
    int64_t HW = H * W;
    auto y = torch::zeros({B, C_out, H, W}, x.options());
    for (int64_t b = 0; b < B; ++b) {
        // x[b] is (C_in, H*W) row-major view
        // w    is (C_out, C_in)
        // y[b] is (C_out, H*W) row-major view
        Kuiper_GEMM_BlockTiling2D_g_gemm_f32_128x128x32_8x8(
            1.0f, 0.0f,
            (uint32_t)C_out, (uint32_t)C_in, (uint32_t)HW,
            w.data_ptr<float>(),
            x[b].data_ptr<float>(),
            y[b].data_ptr<float>());
    }
    return y;
}
```

(Padding logic copied from challenge1 if any of `C_out`, `C_in`,
`H*W` fall short of the GEMM tile minimums; for #87 they don't.)

**Orchestrator F* module** (`Kuiper.KB.PointwiseConv2D` —
~150 LOC):

  - Pulse `fn` that loops `b = 0..B-1` calling the verified
    `Kuiper.Kernel.GEMM.BlockTiling2D` per batch with the appropriate
    sub-array views.
  - Per-batch postcondition is `__matmul_single (transpose_view weight)
    (slice x b)`, which matches `pwconv2d_single x weight bias b oc oh
    ow` once we expose the equivalence
    `__matmul_single ≡ __pwconv2d_single` via index arithmetic
    (one inductive lemma).

Estimated impl LOC: ~150 in F* + ~50 in CUDA bridge + ~25 in
Python + ~5 in `run.sh`.  No new kernel proofs needed.

### 3.2 #82–#85 (depthwise) — WAITS ON `Conv2D.Naive` KERNEL

Depthwise can be implemented as a specialised `Conv2D.Naive` with
the input-channel inner loop dropped.  At present **no `Conv2D.Naive`
kernel has landed** in tree — `Kuiper.Spec.Conv2D` is in but no
device-side primitive for standard 2D convolution exists yet
(see `src/kernelbench/level1/challenge50/STATUS.txt`).

**Resume plan once `Conv2D.Naive` lands:**

  1. Specialise the Conv2D kernel by setting `C_in = 1` per group and
     iterating per channel — concretely, write
     `Kuiper.Kernel.DepthwiseConv2D.Naive.fst` mirroring
     `Conv2D.Naive` but indexing weight as `weight[c, kh, kw]`
     (no `ic` loop) and input as `x[b, c, oh*S+kh*D-P, ow*S+kw*D-P]`.
  2. Per-thread proof: each output thread computes one
     `dwconv2d_single` value — straightforward inductive invariant
     over the linearised `(kh, kw)` index, mirroring the proof
     structure of `Conv2D.Naive` minus the `ic` dimension.
  3. Setup / teardown: split read permission on `x[b, c, :, :]` over
     the `H_out * W_out` per-thread fractions per channel.  Each
     input pixel `x[b, c, h, w]` is read by at most `K_h * K_w`
     output threads → `K_h*K_w`-fold `forevery_factor'` plumbing
     (well within reach; analogous to RowScale's split).

Per-channel parallelism is embarrassingly parallel — depthwise is
strictly easier than full Conv2D because each channel's computation
is independent.  Estimated impl LOC: ~600 for the kernel + setup
+ teardown, ~150 per per-challenge wrapper (×4 → ~600), ~80 per
bridge + Python + run.sh (×4 → ~320).

#82 is the natural first target (fully square, no asymmetry, no
dilation, no padding); generalising to #83/#84/#85 is mostly
parameterisation.

### 3.3 #86 (separable) — COMPOSITION

Once #82 (depthwise) and #87 (pointwise) ship, #86 is the obvious
composition:

```
y_mid = depthwise_kernel(x, dw_w)
y     = pointwise_kernel(y_mid, pw_w)
```

The compositional spec `Kuiper.Spec.SeparableConv2D.separable_conv2d`
is already in tree.  Estimated impl LOC: ~100 for the orchestrator
F* module + ~80 for bridge / Python / run.sh.

#86 must wait on **both** #82 and #87.  In the meantime its
`STATUS.txt` will point at this design doc and the two upstream
dependencies.

---

## 4. LOC estimates

| Component                                     | LOC   | Status                             |
|:----------------------------------------------|:-----:|:-----------------------------------|
| `Kuiper.Spec.DepthwiseConv2D` (`fst`+`fsti`)  | ~250  | ✅ landed                          |
| `Kuiper.Spec.PointwiseConv2D` (`fst`+`fsti`)  | ~190  | ✅ landed                          |
| `Kuiper.Spec.SeparableConv2D` (`fst` only)    |  ~85  | ✅ landed                          |
| `Kuiper.KB.PointwiseConv2D` orchestrator      | ~150  | ❌ Phase 1 (#87, unblocked)        |
| #87 bridge + Python + run.sh                  |  ~80  | ❌ Phase 1                         |
| `Kuiper.Kernel.DepthwiseConv2D.Naive`         | ~600  | ❌ blocked on `Conv2D.Naive`       |
| Per-challenge `Kuiper.KB.DepthwiseConv2D.*`   | ~150 each | ❌ blocked                     |
| #82–#85 bridges + Python + run.sh             |  ~80 each | ❌ blocked                     |
| `Kuiper.KB.SeparableConv2D` orchestrator      | ~100  | ❌ blocked on #82 + #87            |
| #86 bridge + Python + run.sh                  |  ~80  | ❌ blocked                         |

---

## 5. Resume order

**#87 first** — pointwise reduces cleanly to GEMM; no new kernel
required, and it exercises the orchestrator-on-top-of-existing-kernel
pattern that #86 will reuse later.  Per-batch matmul loop in the
bridge is the only "new" plumbing.

**Then #82** (square depthwise) once `Conv2D.Naive` lands.  Use it as
the template for the depthwise kernel proof.  Generalise to #83 / #84
/ #85 after #82 is green.

**Then #86** as a composition of the now-working depthwise and
pointwise orchestrators.  No new verification work besides chaining
the two postconditions via `Kuiper.Spec.SeparableConv2D.separable_conv2d`.

---

## 6. Why this was triaged out of the current session

Phase-0 specs took ~1h of the 2h wall budget.  Shipping #87
end-to-end requires:

  - one orchestrator F* module verifying that "loop the GEMM kernel
    over `B` batches" produces the spec'd `pwconv2d` output;
  - a CUDA bridge with a per-batch loop;
  - the equivalence lemma `__matmul_single ≡ __pwconv2d_single`
    (one ~30-line inductive proof connecting the two left-fold
    accumulators).

This is doable but tight against the wall.  Per the project rules
("if it's >>2h, branch + STATUS.txt + stop"), #87 Phase 1 is left as
the next session's first task.  The four depthwise challenges
(#82–#85) and the separable challenge (#86) are blocked on upstream
kernels and cannot be unblocked in this session.
