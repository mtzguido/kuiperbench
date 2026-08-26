#!/bin/bash
# Test KernelBench L1 #6: Matmul with large K
set -e
cd "$(dirname "$0")/../../../.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${KUIPER_PYTHON:-python3}" src/kernelbench/run_test.py 1 6
