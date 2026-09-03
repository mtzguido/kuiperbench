// Checked bridge for the self-allocating verified L1 normalization entry.
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_L1Norm.h"
#include "Kuiper_KB_L1Norm.cu"

torch::Tensor kuiper_l1norm_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.dim() == 2 &&
                    X.scalar_type() == torch::kFloat32 && X.is_contiguous(),
                "kuiper_l1norm: expected contiguous 2D CUDA float32 tensor");
    int64_t B = X.size(0), D = X.size(1);
    // Kernel requires B > 0, D > 0 and B*D <= max_blocks*max_threads = 2^31.
    TORCH_CHECK(B > 0 && D > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && (__int128) B * D <= ((__int128) 1 << 31),
                "kuiper_l1norm: shape out of range");
    const at::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_L1Norm_l1norm_alloc_f32(
        (uint32_t)B, (uint32_t)D, X.data_ptr<float>());
    return torch::from_blob(out, X.sizes(), [](void *p) { cudaFree(p); },
                            X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_l1norm", &kuiper_l1norm_cuda,
          "Kuiper verified out-of-place L1 normalization along dim=1");
}
