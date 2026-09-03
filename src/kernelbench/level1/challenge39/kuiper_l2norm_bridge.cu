// Checked bridge for the self-allocating verified L2 normalization entry.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_L2Norm.h"
#include "Kuiper_KB_L2Norm.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_l2norm_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.dim() == 2 &&
                    X.scalar_type() == torch::kFloat32 && X.is_contiguous(),
                "kuiper_l2norm: expected contiguous 2D CUDA float32 tensor");
    int64_t B = X.size(0), D = X.size(1);
    TORCH_CHECK(B > 0 && D > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && (__int128) B * D <= KUIPER_MAX_NTHR
                && D + 1024 <= (int64_t)UINT32_MAX,
                "kuiper_l2norm: shape out of range");
    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_L2Norm_l2norm_alloc_f32(
        (uint32_t)B, (uint32_t)D, X.data_ptr<float>());
    return torch::from_blob(out, X.sizes(), [](void *p) { cudaFree(p); },
                            X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_l2norm", &kuiper_l2norm_cuda,
          "Kuiper verified out-of-place L2 normalization along dim=1");
}
