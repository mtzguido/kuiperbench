#!/bin/bash
# Test KernelBench L1 #87: pointwise (1x1) 2D conv forward
set -e
cd "$(dirname "$0")/../../../.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${KUIPER_PYTHON:-python3}" src/kernelbench/run_test.py 1 87
