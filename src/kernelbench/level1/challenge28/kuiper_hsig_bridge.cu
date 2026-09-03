// Bridge for KernelBench L1: HardSigmoid.
// Kuiper allocates and computes the result from the original input.

#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_HardSigmoid.h"
#include "Kuiper_KB_HardSigmoid.cu"

torch::Tensor kuiper_hsig_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(),
                "kuiper #28: expected a contiguous CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 ||
                    X.scalar_type() == torch::kFloat64,
                "kuiper_hsig: unsupported dtype");
    int64_t numel = X.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #28: element count exceeds the verified kernel bound");

    const at::cuda::CUDAGuard device_guard(X.device());
    if (X.scalar_type() == torch::kFloat32) {
        float *output = Kuiper_KB_HardSigmoid_hsig_alloc_f32(
            (uint32_t)numel, X.data_ptr<float>());
        return torch::from_blob(
            output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
    }
    double *output = Kuiper_KB_HardSigmoid_hsig_alloc_f64(
        (uint32_t)numel, X.data_ptr<double>());
    return torch::from_blob(
        output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_hsig", &kuiper_hsig_cuda, "Kuiper verified hsig");
}
