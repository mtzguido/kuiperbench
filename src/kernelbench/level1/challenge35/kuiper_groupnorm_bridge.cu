// Checked PyTorch boundary for KernelBench L1 #35.  The bridge validates the
// raw NCHW/group contract and makes one call; C/G, row geometry, allocation,
// input copy, and normalization are composed inside Kuiper.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

#include "Kuiper_KB_MeanVarNorm.h"
#include "Kuiper_KB_MeanVarNorm.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_groupnorm_cuda(torch::Tensor X, int64_t G, double eps) {
    TORCH_CHECK(X.is_cuda() && X.dim() == 4 &&
                    X.scalar_type() == torch::kFloat32,
                "kuiper_groupnorm: expected 4D CUDA float32 tensor");
    TORCH_CHECK(X.is_contiguous(),
                "kuiper_groupnorm: input must be contiguous");

    const int64_t B = X.size(0), C = X.size(1);
    const int64_t H = X.size(2), W = X.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0 && G > 0 && G <= C &&
                    B <= (int64_t) UINT32_MAX &&
                    C <= (int64_t) UINT32_MAX &&
                    H <= (int64_t) UINT32_MAX &&
                    W <= (int64_t) UINT32_MAX &&
                    G <= (int64_t) UINT32_MAX && C % G == 0 &&
                    X.numel() <= KUIPER_MAX_NTHR,
                "kuiper_groupnorm: shape/groups are outside the verified range");

    TORCH_CHECK(std::isfinite(eps) &&
                    eps >= std::numeric_limits<float>::denorm_min() &&
                    eps <= std::numeric_limits<float>::max(),
                "kuiper_groupnorm: eps must fit the positive float32 range");

    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_MeanVarNorm_groupnorm35_alloc_f32(
        (uint32_t) B, (uint32_t) C, (uint32_t) H, (uint32_t) W,
        (uint32_t) G, eps, X.data_ptr<float>());
    return torch::from_blob(out, {B, C, H, W},
                            [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_groupnorm", &kuiper_groupnorm_cuda,
          "Kuiper verified GroupNorm (identity affine)");
}
