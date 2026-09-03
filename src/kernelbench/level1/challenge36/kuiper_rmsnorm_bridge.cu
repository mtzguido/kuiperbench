// Bridge for KernelBench L1 #36: RMSNorm along dim=1.
//
// Input X has shape (B, C, H, W).  RMSNorm divides by
//     rms[b,h,w] = sqrt( mean_c X[b,c,h,w]^2 + eps )
//
// The exported Kuiper entry derives the spatial view, allocates/copies the
// result, and performs the complete normalization under one top-level spec.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>
#include "Kuiper_KB_RMSNorm.h"
#include "Kuiper_KB_RMSNorm.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_rmsnorm_cuda(torch::Tensor X, double eps) {
    TORCH_CHECK(X.is_cuda() && X.dim() == 4 &&
                    X.scalar_type() == torch::kFloat32 && X.is_contiguous(),
                "kuiper_rmsnorm: expected contiguous 4D CUDA float32 tensor");
    int64_t B = X.size(0), C = X.size(1), H = X.size(2), W = X.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0 &&
                    B <= (int64_t)UINT32_MAX && C <= (int64_t)UINT32_MAX &&
                    H <= (int64_t)UINT32_MAX && W <= (int64_t)UINT32_MAX &&
                    X.numel() <= KUIPER_MAX_NTHR,
                "kuiper_rmsnorm: shape out of range");
    TORCH_CHECK(std::isfinite(eps) &&
                    eps >= std::numeric_limits<float>::denorm_min() &&
                    eps <= std::numeric_limits<float>::max(),
                "kuiper_rmsnorm: eps must fit the positive float32 range");
    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_RMSNorm_rmsnorm4d_alloc_f32(
        (uint32_t)B, (uint32_t)C, (uint32_t)H, (uint32_t)W,
        eps, X.data_ptr<float>());
    return torch::from_blob(out, X.sizes(), [](void *p) { cudaFree(p); },
                            X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_rmsnorm", &kuiper_rmsnorm_cuda,
          "Kuiper verified out-of-place RMSNorm along dim=1");
}
