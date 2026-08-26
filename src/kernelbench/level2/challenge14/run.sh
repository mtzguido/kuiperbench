#!/bin/bash
# Test KernelBench L2 #14: Gemm_Divide_Sum_Scaling
set -e
cd "$(dirname "$0")/../../../.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${KUIPER_PYTHON:-python3}" src/kernelbench/run_test.py 2 14
