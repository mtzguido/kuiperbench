# KuiperBench

KuiperBench is a collection of verified CUDA solutions for
[KernelBench](https://github.com/ScalingIntelligence/KernelBench), written in
[Kuiper](https://github.com/FStarLang/kuiper).

Kuiper builds on F\* and Pulse, using dependent types and separation logic to
make GPU programs memory-safe and data-race-free by construction. The Kuiper
modules in this repository also state and prove functional postconditions for
the implemented kernels. GPU comparison against the PyTorch reference models
provides additional experimental validation; it does not replace the proofs.

The formal basis is described in
[Kuiper: Correct and Efficient GPU Programming with Dependent Types and Separation Logic](https://doi.org/10.1145/3808280)
(PLDI 2026). The motivation for applying these guarantees to generated GPU
kernels is developed in
[The Next Frontier for AI-Generated Kernels: Correctness](https://doi.org/10.1145/3819802.3820580)
(PAgE 2026).

## Verification scope

`make verify` checks every local F\*/Pulse module against a fixed Kuiper release
without admitting SMT queries. `make list-admits` reports the remaining explicit
trust markers so that they can be reviewed directly.

The formal claims rely on the usual trusted computing base: F\*, Pulse, Z3,
Karamel extraction, the C/CUDA compiler and runtime, the small handwritten
Python/C++ interface, and the explicitly documented compatibility primitives.
The generated CUDA in `dist/` is derived from the verified modules, and CI
checks that it agrees with a fresh extraction.

KuiperBench is verified independently against a fixed Kuiper package. The
repository therefore contains only the definitions specific to this project;
the package supplies the corresponding Kuiper library, F\*, Pulse, Karamel,
Z3, extraction plugin, and CUDA headers.

## Reproducing the verification

The selected Kuiper nightly is recorded in the root-level `kuiper-version.txt` file.
The first command retrieves it into the ignored `.kuiper/` directory; later
checks reuse the same version.

Extraction uses clang-format 19.1.7, retrieved into the ignored `.tools/`
directory. Fixing the formatter version and configuration makes the generated
`dist/` snapshot reproducible across machines.

```bash
make -j$(nproc) prepare
make -j$(nproc)
```

Useful targets are:

```bash
make -j$(nproc) verify       # check all local F*/Pulse modules
make -j$(nproc) extract-all  # generate CUDA headers and sources in obj/
make -j$(nproc) dist         # refresh the checked-in dist/ output
make -j$(nproc) test-kb-1    # compare one solution with KernelBench on a GPU
make -j$(nproc) test-kb-all  # compare every solution with KernelBench on a GPU
```

## GPU testing

Continuous integration performs formal verification, extraction, and a check
that the generated `dist/` files are current. It does not compile or execute
the kernels on a GPU.

Researchers can run the experimental comparisons on their own CUDA-equipped
systems. Initialize the KernelBench submodule, then run one challenge or the
complete collection:

```bash
git submodule update --init KernelBench
make -j$(nproc) test-kb-1
make -j$(nproc) test-kb-all
```

The complete collection may require substantial GPU memory, depending on the
KernelBench inputs. These comparisons complement the formal results described
above; they are not part of the repository's CI guarantee.

To check against an existing Kuiper package, set `KUIPER_HOME`:

```bash
make -j$(nproc) KUIPER_HOME=/opt/kuiper verify
```

## Repository structure

- `src/lib/`: reusable specifications, kernels, data layouts, and explicit
  compatibility modules belonging to KuiperBench
- `src/kernelbench/`: monomorphic KernelBench entry points and Python loaders
- `src/examples/` and `src/klas/`: standalone extracted entry points
- `include/kbench.h`: local CUDA support used at the extraction boundary
- `dist/`: reproducible CUDA generated from the verified sources

New definitions use distinct module names, normally under `Kuiper.KB`. When a
compatibility primitive becomes part of Kuiper itself, KuiperBench can use the
released definition and remove its local trusted counterpart.
