// Bridge for KernelBench L1 #21: Sigmoid.
// Kuiper allocates and computes the result from the original input.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_Sigmoid.h"
#include "Kuiper_KB_Sigmoid.cu"

torch::Tensor kuiper_sigmoid_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(),
                "kuiper #21: expected a contiguous CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 ||
                    X.scalar_type() == torch::kFloat64,
                "kuiper_sigmoid: unsupported dtype");
    int64_t numel = X.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #21: element count exceeds the verified kernel bound");

    const c10::cuda::CUDAGuard device_guard(X.device());
    if (X.scalar_type() == torch::kFloat32) {
        float *output = Kuiper_KB_Sigmoid_sigmoid_alloc_f32(
            (uint32_t)numel, X.data_ptr<float>());
        return torch::from_blob(
            output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
    }
    double *output = Kuiper_KB_Sigmoid_sigmoid_alloc_f64(
        (uint32_t)numel, X.data_ptr<double>());
    return torch::from_blob(
        output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_sigmoid", &kuiper_sigmoid_cuda, "Kuiper verified sigmoid");
}
