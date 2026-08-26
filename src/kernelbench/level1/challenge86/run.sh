#!/bin/bash
# Test KernelBench L1 #86: depthwise-separable 2D conv (dw + pw)
set -e
cd "$(dirname "$0")/../../../.."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${KUIPER_PYTHON:-python3}" src/kernelbench/run_test.py 1 86
