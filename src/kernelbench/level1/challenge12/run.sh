#!/bin/bash
# Test KernelBench L1 #12: Matmul with diagonal (diag(A) @ B)
set -e
cd "$(dirname "$0")/../../../.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${KUIPER_PYTHON:-python3}" src/kernelbench/run_test.py 1 12
