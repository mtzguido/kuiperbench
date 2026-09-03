// Bridge for KernelBench L1 #22: Tanh.
// Kuiper allocates and computes the result from the original input.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_Tanh.h"
#include "Kuiper_KB_Tanh.cu"

torch::Tensor kuiper_tanh_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(),
                "kuiper #22: expected a contiguous CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 ||
                    X.scalar_type() == torch::kFloat64,
                "kuiper_tanh: unsupported dtype");
    int64_t numel = X.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #22: element count exceeds the verified kernel bound");

    const c10::cuda::CUDAGuard device_guard(X.device());
    if (X.scalar_type() == torch::kFloat32) {
        float *output = Kuiper_KB_Tanh_tanh_alloc_f32(
            (uint32_t)numel, X.data_ptr<float>());
        return torch::from_blob(
            output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
    }
    double *output = Kuiper_KB_Tanh_tanh_alloc_f64(
        (uint32_t)numel, X.data_ptr<double>());
    return torch::from_blob(
        output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_tanh", &kuiper_tanh_cuda, "Kuiper verified tanh");
}
