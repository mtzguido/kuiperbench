#!/usr/bin/env python3
"""Compile and load every KernelBench solution's PyTorch extension."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS_ROOT = ROOT / "src" / "kernelbench"


def numbered(path: Path, prefix: str) -> int:
    name = path.name
    if not name.startswith(prefix):
        raise ValueError(f"expected {prefix!r} prefix in {path}")
    return int(name.removeprefix(prefix))


def find_solutions() -> list[tuple[int, int, Path]]:
    solutions: list[tuple[int, int, Path]] = []
    errors: list[str] = []

    level_dirs = sorted(
        SOLUTIONS_ROOT.glob("level*"), key=lambda path: numbered(path, "level")
    )
    for level_dir in level_dirs:
        level = numbered(level_dir, "level")
        challenge_dirs = sorted(
            level_dir.glob("challenge*"),
            key=lambda path: numbered(path, "challenge"),
        )
        for challenge_dir in challenge_dirs:
            challenge = numbered(challenge_dir, "challenge")
            candidates = sorted(
                path
                for path in challenge_dir.glob("*.py")
                if not path.name.startswith("_")
            )
            if len(candidates) != 1:
                errors.append(
                    f"{challenge_dir.relative_to(ROOT)}: expected one solution, "
                    f"found {len(candidates)}"
                )
                continue
            solutions.append((level, challenge, candidates[0]))

    if errors:
        raise RuntimeError("\n".join(errors))
    return solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list without building")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    return args


def main() -> int:
    args = parse_args()
    solutions = find_solutions()
    selected = solutions[args.shard_index :: args.shard_count]

    if args.list:
        for level, challenge, solution in selected:
            print(f"L{level} #{challenge}: {solution.relative_to(ROOT)}")
        print(f"{len(selected)}/{len(solutions)} challenge(s)")
        return 0

    import torch

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}", flush=True)
    print(f"CUDA device available: {torch.cuda.is_available()}", flush=True)
    print(
        f"Building shard {args.shard_index + 1}/{args.shard_count}: "
        f"{len(selected)} of {len(solutions)} challenge(s)",
        flush=True,
    )

    for position, (level, challenge, solution) in enumerate(selected, start=1):
        relative = solution.relative_to(ROOT)
        print(
            f"[{position}/{len(selected)}] L{level} #{challenge}: {relative}",
            flush=True,
        )
        result = subprocess.run([sys.executable, str(solution)], cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED: {relative}", file=sys.stderr)
            return result.returncode

    print(f"Built and loaded {len(selected)} challenge extension(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
