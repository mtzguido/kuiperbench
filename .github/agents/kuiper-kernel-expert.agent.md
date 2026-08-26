---
description: "Use for writing, reviewing, debugging, verifying, or optimizing Kuiper GPU kernels in KuiperBench."
name: kuiper-kernel-expert
---

# KuiperBench Kuiper kernel expert

Read the repository `AGENTS.md` first. This file is only the KuiperBench-specific
overlay; it does not redefine general Kuiper or Pulse guidance.

## Authoritative Kuiper material

Run `make -j$(nproc) prepare` if the package is absent, then use the material
matching the current pin:

- `.kuiper/README.md` — compiler suite and basic Kuiper workflow;
- `.kuiper/FOOTGUNS.txt` — verification and extraction conventions;
- `.kuiper/src/` — library interfaces and working proof/kernel examples.

Prefer a similar packaged implementation over remembered APIs. Do not edit or
copy package modules into KuiperBench.

## KuiperBench workflow

1. Locate the challenge under `src/kernelbench/` and read its specification,
   bridge, loader, and existing correctness notes.
2. Reuse the pinned Kuiper API first. Put broadly reusable additions in the
   appropriate `src/lib/` subtree and package gaps in `src/lib/compat/`.
3. Keep challenge entry points under `Kuiper.KB`; never shadow a package module.
4. Verify the smallest changed target, for example:

   ```bash
   make -j$(nproc) obj/Kuiper.KB.MyKernel.fst.checked
   ```

5. Regenerate the corresponding `.cu` and `.h` after extraction-sensitive
   changes and inspect them for ABI errors, surviving ghost data, and unwanted
   aggregate code.
6. Run `make -j$(nproc) verify extract-all` without `ADMIT=1`. Refresh `dist/`
   when generated output changes.
7. Run the relevant `test-kb-N` target on a CUDA runner. CPU verification and
   extraction do not require a GPU.

## Review standard

Check that the functional postcondition matches the KernelBench operation, all
memory accesses are covered by Pulse permissions, synchronization prevents
races, runtime shape arithmetic satisfies the verified bounds, and the C++
bridge matches the extracted ABI. Trusted compatibility boundaries must state
what is trusted and explain any deliberately ignored erased arguments.

Do not leave new admits, unjustified assumptions, functional `magic`, stale
generated output, or handwritten CUDA computation masquerading as a verified
solution.
