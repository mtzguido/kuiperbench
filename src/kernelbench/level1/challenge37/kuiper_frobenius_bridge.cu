// Checked bridge for the self-allocating verified FrobeniusNorm entry.
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_Frobenius.h"
#include "Kuiper_KB_Frobenius.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_frobenius_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 &&
                    X.is_contiguous(),
                "kuiper_frobenius: expected contiguous CUDA float32 tensor");
    int64_t numel = X.numel();
    TORCH_CHECK(numel > 0 && numel <= KUIPER_MAX_NTHR,
                "kuiper_frobenius: numel exceeds verified launch range");
    const at::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_Frobenius_frobenius_alloc_f32(
        (uint32_t)numel, X.data_ptr<float>());
    return torch::from_blob(out, X.sizes(), [](void *p) { cudaFree(p); },
                            X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_frobenius", &kuiper_frobenius_cuda,
          "Kuiper verified out-of-place Frobenius norm normalization");
}
