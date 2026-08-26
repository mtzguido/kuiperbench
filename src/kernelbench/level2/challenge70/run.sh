#!/bin/bash
# Test KernelBench L2 #70: Gemm_Sigmoid_Scaling_ResidualAdd
set -e
cd "$(dirname "$0")/../../../.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${KUIPER_PYTHON:-python3}" src/kernelbench/run_test.py 2 70
