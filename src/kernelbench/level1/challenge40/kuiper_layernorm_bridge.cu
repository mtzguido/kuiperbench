// Bridge for KernelBench L1 #40: LayerNorm.
//
// PyTorch's nn.LayerNorm(normalized_shape) reduces over the last
// len(normalized_shape) dims; for the KernelBench harness the input is
// (batch_size=16, features=64, dim1=256, dim2=256) and normalized_shape
// is (features, dim1, dim2), so the reduction collapses everything past
// dim 0.  We view that as Array1 of shape (B, n) with n = features*dim1*dim2,
// flat row-major.  Per row r:
//   mean[r]    = (1/n) Σ_j x[r,j]
//   inv_std[r] = 1 / sqrt((1/n) Σ_j x[r,j]^2 - mean[r]^2 + eps)
//   y[r,j]     = ((x[r,j] - mean[r]) * inv_std[r]) * γ[j] + β[j]
//
// The exported Kuiper entry derives n=C*H*W, allocates/copies the result,
// and runs the complete operation under one top-level specification.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>
#include "Kuiper_KB_LayerNorm.h"
#include "Kuiper_KB_LayerNorm.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_layernorm_cuda(torch::Tensor X, torch::Tensor gamma,
                                    torch::Tensor beta, double eps) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda() &&
                    X.device() == gamma.device() && X.device() == beta.device(),
                "kuiper_layernorm: all tensors must share a CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                gamma.scalar_type() == torch::kFloat32 &&
                beta.scalar_type() == torch::kFloat32,
                "kuiper_layernorm: all tensors must be float32");
    TORCH_CHECK(X.dim() == 4 && gamma.dim() == 3 && beta.dim() == 3 &&
                    X.is_contiguous() && gamma.is_contiguous() &&
                    beta.is_contiguous(),
                "kuiper_layernorm: expected contiguous X[B,C,H,W] and affine[C,H,W]");
    int64_t B = X.size(0), C = X.size(1), H = X.size(2), W = X.size(3);
    TORCH_CHECK(gamma.size(0) == C && gamma.size(1) == H &&
                    gamma.size(2) == W && beta.sizes() == gamma.sizes(),
                "kuiper_layernorm: gamma/beta shape mismatch");
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0 &&
                    B <= (int64_t)UINT32_MAX && C <= (int64_t)UINT32_MAX &&
                    H <= (int64_t)UINT32_MAX && W <= (int64_t)UINT32_MAX &&
                    X.numel() <= KUIPER_MAX_NTHR,
                "kuiper_layernorm: shape out of range");
    TORCH_CHECK(std::isfinite(eps) &&
                    eps >= std::numeric_limits<float>::denorm_min() &&
                    eps <= std::numeric_limits<float>::max(),
                "kuiper_layernorm: eps must fit the positive float32 range");
    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_LayerNorm_layernorm4d_alloc_f32(
        (uint32_t)B, (uint32_t)C, (uint32_t)H, (uint32_t)W, eps,
        X.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>());
    return torch::from_blob(out, X.sizes(), [](void *p) { cudaFree(p); },
                            X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_layernorm", &kuiper_layernorm_cuda,
          "Kuiper verified out-of-place LayerNorm");
}
