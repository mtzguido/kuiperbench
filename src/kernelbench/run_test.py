#!/usr/bin/env python3
"""
Run KernelBench correctness tests for Kuiper solutions.

Thin wrapper over ``kernelbench.eval.eval_kernel_against_ref`` — we do NOT
reimplement the comparison, tolerances, trials, or timing logic.  If you
find yourself wanting to hand-roll anything here, don't: fix KernelBench
upstream or live with their defaults.

Usage:
    ./src/kernelbench/run_test.py 1 5       # test level 1, challenge 5
    ./src/kernelbench/run_test.py            # test all solved challenges
"""

import sys
import os
import glob

KUIPER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_REPO = os.path.join(KUIPER_ROOT, "KernelBench")
KB_ROOT = os.path.join(KB_REPO, "KernelBench")
SOL_ROOT = os.path.join(KUIPER_ROOT, "src", "kernelbench")

# Make `from kernelbench.eval import ...` work without installing the package.
sys.path.insert(0, os.path.join(KB_REPO, "src"))

from kernelbench.eval import eval_kernel_against_ref  # noqa: E402


def find_solution(level, problem_id):
    d = os.path.join(SOL_ROOT, f"level{level}", f"challenge{problem_id}")
    if not os.path.isdir(d):
        return None
    py = glob.glob(os.path.join(d, f"{problem_id}_*.py"))
    if not py:
        py = [p for p in glob.glob(os.path.join(d, "*.py")) if "__" not in os.path.basename(p)]
    return py[0] if py else None


def find_reference(level, problem_id):
    refs = glob.glob(os.path.join(KB_ROOT, f"level{level}", f"{problem_id}_*.py"))
    return refs[0] if refs else None


def _read_solution_src(sol_path):
    """Read solution source and inject __file__ so relative-path loading works
    when the source is exec()'d inside KernelBench's load_custom_model."""
    with open(sol_path) as f:
        body = f.read()
    header = f"__file__ = {sol_path!r}\n"
    return header + body


def run_test(level, problem_id):
    ref_path = find_reference(level, problem_id)
    if not ref_path:
        print(f"  SKIP: no reference for level{level}/{problem_id}")
        return None
    sol_path = find_solution(level, problem_id)
    if not sol_path:
        print(f"  SKIP: no solution for level{level}/challenge{problem_id}")
        return None

    print(f"  Testing L{level} #{problem_id}: {os.path.basename(ref_path)[:-3]}")

    with open(ref_path) as f:
        ref_src = f.read()
    sol_src = _read_solution_src(sol_path)

    result = eval_kernel_against_ref(
        original_model_src=ref_src,
        custom_model_src=sol_src,
        num_correct_trials=1,
        measure_performance=False,
        verbose=False,
    )

    if result is None:
        print("    RESULT: lock/retry — no verdict")
        return None
    status = "PASS" if result.correctness else ("FAIL" if result.compiled else "COMPILE-FAIL")
    meta = result.metadata or {}
    extra = ""
    if "max_difference" in meta:
        extra = f", max_diff={meta['max_difference']}"
    elif "correctness_issue" in meta:
        extra = f", issue={meta['correctness_issue']}"
    print(f"    {status}{extra}")
    return bool(result.correctness)


def main():
    if len(sys.argv) == 3:
        level, pid = int(sys.argv[1]), int(sys.argv[2])
        ok = run_test(level, pid)
        sys.exit(0 if ok else 1)
    elif len(sys.argv) == 1:
        import subprocess
        results = []
        for level_dir in sorted(glob.glob(os.path.join(SOL_ROOT, "level*"))):
            level = int(os.path.basename(level_dir).replace("level", ""))
            for challenge_dir in sorted(glob.glob(os.path.join(level_dir, "challenge*"))):
                pid = int(os.path.basename(challenge_dir).replace("challenge", ""))
                if not find_solution(level, pid):
                    continue
                r = subprocess.run(
                    [sys.executable, __file__, str(level), str(pid)],
                    capture_output=True, text=True, env=dict(os.environ),
                )
                print(r.stdout.strip())
                ok = r.returncode == 0
                results.append((level, pid, ok))
        passed = sum(1 for _, _, ok in results if ok)
        print(f"\nResults: {passed}/{len(results)} passed")
        sys.exit(0 if passed == len(results) else 1)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
