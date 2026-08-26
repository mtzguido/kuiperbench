#!/bin/bash
# Test KernelBench L1 #92: exclusive cumulative sum along dim=1
set -e
cd "$(dirname "$0")/../../../.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${KUIPER_PYTHON:-python3}" src/kernelbench/run_test.py 1 92
