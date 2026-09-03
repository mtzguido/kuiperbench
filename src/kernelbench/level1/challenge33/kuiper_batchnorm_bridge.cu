// Bridge for KernelBench L1 #33: BatchNorm2d.
//
// PyTorch's nn.BatchNorm2d(num_features=C) operates on (N, C, H, W)
// row-major float32; in default training mode it normalizes per-channel
// using batch statistics (biased variance, denom = N*H*W).
//
// We view (N, C, H, W) as the strided (C, N*HW) matrix via the verified
// Kuiper.Tensor.Layout.BCMChannels layout, so "row ci" of the matrix is
// the entirety of channel ci across all batch and spatial positions.
// For each channel ci:
//   mean[ci]  = (1/(N*H*W)) Σ x[ci,k]
//   var[ci]   = (1/(N*H*W)) Σ x[ci,k]^2 - mean[ci]^2
//   y[ci,k]   = (x[ci,k] - mean[ci]) / sqrt(var[ci] + eps) * γ[ci] + β[ci]
//
// γ and β have shape (C,); they are read with fractional permission.
// The exported Kuiper entry derives all flattened geometry, allocates/copies
// the result, and orchestrates the full operation under one top-level spec.
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>
#include "Kuiper_KB_BatchNorm.h"
#include "Kuiper_KB_BatchNorm.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_batchnorm_cuda(torch::Tensor X, torch::Tensor gamma,
                                    torch::Tensor beta, double eps) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "kuiper_batchnorm: all tensors must be CUDA");
    TORCH_CHECK(X.device() == gamma.device() && X.device() == beta.device(),
                "kuiper_batchnorm: all tensors must be on the same device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                gamma.scalar_type() == torch::kFloat32 &&
                beta.scalar_type() == torch::kFloat32,
                "kuiper_batchnorm: all tensors must be float32");
    TORCH_CHECK(X.dim() == 4 && X.is_contiguous() && gamma.is_contiguous() &&
                    beta.is_contiguous(),
                "kuiper_batchnorm: tensors must be contiguous");
    int64_t N = X.size(0);
    int64_t C = X.size(1);
    int64_t H = X.size(2);
    int64_t W = X.size(3);
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1 &&
                    gamma.size(0) == C && beta.size(0) == C,
                "kuiper_batchnorm: gamma/beta numel must equal C");
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0 &&
                    N <= (int64_t)UINT32_MAX && C <= (int64_t)UINT32_MAX &&
                    H <= (int64_t)UINT32_MAX && W <= (int64_t)UINT32_MAX &&
                    X.numel() <= KUIPER_MAX_NTHR,
                "kuiper_batchnorm: shape out of range");
    TORCH_CHECK(std::isfinite(eps) &&
                    eps >= std::numeric_limits<float>::denorm_min() &&
                    eps <= std::numeric_limits<float>::max(),
                "kuiper_batchnorm: eps must fit the positive float32 range");
    const at::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_BatchNorm_batchnorm2d_alloc_f32(
        (uint32_t)N, (uint32_t)C, (uint32_t)H, (uint32_t)W, eps,
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>());
    return torch::from_blob(out, X.sizes(), [](void *p) { cudaFree(p); },
                            X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_batchnorm", &kuiper_batchnorm_cuda,
          "Kuiper verified out-of-place BatchNorm2d");
}
