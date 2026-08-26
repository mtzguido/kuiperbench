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
// The verified entry point [Kuiper.KB.LayerNorm.layernorm_fw_f32] runs
// per-row, in place; γ and β are taken in with fractional permission.
#include <torch/extension.h>
#include "Kuiper_KB_LayerNorm.h"
#include "Kuiper_KB_LayerNorm.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_layernorm_cuda(torch::Tensor X, torch::Tensor gamma,
                                    torch::Tensor beta, double eps) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "kuiper_layernorm: all tensors must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                gamma.scalar_type() == torch::kFloat32 &&
                beta.scalar_type() == torch::kFloat32,
                "kuiper_layernorm: all tensors must be float32");
    TORCH_CHECK(X.dim() >= 2, "kuiper_layernorm: X must be at least 2D");
    auto Xc = X.contiguous();
    auto Gc = gamma.contiguous();
    auto Bc = beta.contiguous();
    int64_t B = Xc.size(0);
    int64_t n = 1;
    for (int i = 1; i < Xc.dim(); ++i) n *= Xc.size(i);
    TORCH_CHECK(Gc.numel() == n && Bc.numel() == n,
                "kuiper_layernorm: gamma/beta numel mismatch");
    TORCH_CHECK(B > 0 && n > 0
                && B <= (int64_t)UINT32_MAX
                && n <= (int64_t)UINT32_MAX
                && B * n <= (int64_t)UINT32_MAX
                && B * n <= KUIPER_MAX_NTHR
                && n + 1024 <= (int64_t)UINT32_MAX,
                "kuiper_layernorm: shape out of range");
    Kuiper_KB_LayerNorm_layernorm_fw_f32(
        (uint32_t)B, (uint32_t)n,
        (float)eps,
        Xc.data_ptr<float>(), Gc.data_ptr<float>(), Bc.data_ptr<float>());
    return Xc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_layernorm", &kuiper_layernorm_cuda,
          "Kuiper verified LayerNorm: per-row (x-μ)/σ then γ*+β");
}
