// Checked PyTorch boundary for KernelBench L1 #34.  The bridge validates the
// raw NCHW contract and makes one call; row geometry, allocation, input copy,
// reciprocal construction, and normalization are composed inside Kuiper.
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

#include "Kuiper_KB_MeanVarNorm.h"
#include "Kuiper_KB_MeanVarNorm.cu"

static constexpr int64_t KUIPER_MVN_MAX_NTHR =
    (int64_t) 2097152 * 1024;

torch::Tensor kuiper_instancenorm_cuda(torch::Tensor X, double eps) {
    TORCH_CHECK(X.is_cuda() && X.dim() == 4 &&
                    X.scalar_type() == torch::kFloat32,
                "kuiper_instancenorm: expected 4D CUDA float32 tensor");
    TORCH_CHECK(X.is_contiguous(),
                "kuiper_instancenorm: input must be contiguous");

    const int64_t B = X.size(0), C = X.size(1);
    const int64_t H = X.size(2), W = X.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0 &&
                    B <= (int64_t) UINT32_MAX &&
                    C <= (int64_t) UINT32_MAX &&
                    H <= (int64_t) UINT32_MAX &&
                    W <= (int64_t) UINT32_MAX &&
                    X.numel() <= KUIPER_MVN_MAX_NTHR,
                "kuiper_instancenorm: shape is outside the verified range");

    TORCH_CHECK(std::isfinite(eps) &&
                    eps >= std::numeric_limits<float>::denorm_min() &&
                    eps <= std::numeric_limits<float>::max(),
                "kuiper_instancenorm: eps must fit the positive float32 range");

    const at::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_MeanVarNorm_instancenorm34_alloc_f32(
        (uint32_t) B, (uint32_t) C, (uint32_t) H, (uint32_t) W,
        eps, X.data_ptr<float>());
    return torch::from_blob(out, {B, C, H, W},
                            [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_instancenorm", &kuiper_instancenorm_cuda,
          "Kuiper verified InstanceNorm2d (no affine)");
}
