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
// The verified entry point [Kuiper.KB.BatchNorm.batchnorm_fw_f32]
// orchestrates the per-channel reduction + two-stage affine in place.
#include <torch/extension.h>
#include "Kuiper_KB_BatchNorm.h"
#include "Kuiper_KB_BatchNorm.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_batchnorm_cuda(torch::Tensor X, torch::Tensor gamma,
                                    torch::Tensor beta, double eps) {
    TORCH_CHECK(X.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "kuiper_batchnorm: all tensors must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                gamma.scalar_type() == torch::kFloat32 &&
                beta.scalar_type() == torch::kFloat32,
                "kuiper_batchnorm: all tensors must be float32");
    TORCH_CHECK(X.dim() == 4, "kuiper_batchnorm: X must be 4-D (N,C,H,W)");
    auto Xc = X.contiguous();
    auto Gc = gamma.contiguous();
    auto Bc = beta.contiguous();
    int64_t N = Xc.size(0);
    int64_t C = Xc.size(1);
    int64_t H = Xc.size(2);
    int64_t W = Xc.size(3);
    int64_t HW  = H * W;
    int64_t NHW = N * HW;
    TORCH_CHECK(Gc.numel() == C && Bc.numel() == C,
                "kuiper_batchnorm: gamma/beta numel must equal C");
    TORCH_CHECK(N > 0 && C > 0 && HW > 0
                && C  <= (int64_t)UINT32_MAX
                && HW <= (int64_t)UINT32_MAX
                && NHW <= (int64_t)UINT32_MAX
                && C * HW <= (int64_t)UINT32_MAX
                && N * (HW * C) <= (int64_t)UINT32_MAX
                && NHW <= KUIPER_MAX_NTHR
                && NHW + 1024 <= (int64_t)UINT32_MAX,
                "kuiper_batchnorm: shape out of range");
    Kuiper_KB_BatchNorm_batchnorm_fw_f32(
        (uint32_t)C, (uint32_t)HW, (uint32_t)NHW,
        (float)eps,
        Xc.data_ptr<float>(), Gc.data_ptr<float>(), Bc.data_ptr<float>());
    return Xc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_batchnorm", &kuiper_batchnorm_cuda,
          "Kuiper verified BatchNorm2d: per-channel (x-μ)/σ then γ*+β");
}
