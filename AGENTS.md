# KuiperBench — Agent Instructions

KuiperBench contains verified KernelBench solutions written in Kuiper. It is
checked against a released Kuiper package: never add Kuiper, F*, Pulse, or
Karamel source trees or submodules.

## Kuiper reference

The selected Kuiper package is the authority for general Kuiper usage. After
`make prepare`, consult:

- `.kuiper/README.md` for the included compiler suite and basic workflow;
- `.kuiper/FOOTGUNS.txt` for verification and extraction conventions;
- `.kuiper/src/` for the exact APIs and examples matching the pin.

Do not copy that general guidance into this repository. If `.kuiper/` is not
installed yet, prepare it before doing Kuiper work. For the KuiperBench kernel
workflow, see `.github/agents/kuiper-kernel-expert.agent.md`.

## Build and verify

The Makefile downloads the package into `.kuiper/` and checks only the local
modules against its checked library. Always run Make in parallel:

```bash
make -j$(nproc) prepare
make -j$(nproc) verify
make -j$(nproc) extract-all
make -j$(nproc)
make -j$(nproc) dist
```

Set `KUIPER_HOME=/path/to/package` to use an existing package. `ADMIT=1` is
development-only; final validation must not use it.

GPU correctness tests require CUDA and the KernelBench submodule:

```bash
git submodule update --init KernelBench
make -j$(nproc) test-kb-1
make -j$(nproc) test-kb-all
```

## KuiperBench source boundaries

- `src/lib/compat/` contains temporary bridges missing from the package.
- `src/lib/{data,kernel,kuiper,spec}/` contains reusable KuiperBench modules.
- `src/kernelbench/` contains challenge entry points and Python loaders.
- `include/kbench.h` contains KuiperBench's local CUDA support for generated code.
- `dist/` is checked-in generated output; refresh it with `make dist`.
- `.kuiper/` and `obj/` are generated and must not be committed.

Never shadow or modify a packaged module. Use a distinct local module name,
normally under `Kuiper.KB` for challenge-specific code. When a compatibility
feature reaches the package, delete the local copy and move callers to the
packaged API.
