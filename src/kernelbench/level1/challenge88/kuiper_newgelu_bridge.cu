// Bridge for KernelBench L1 #88: MinGPT NewGELU.
//   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// The fixed constants, allocation, and computation all remain in Kuiper.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_NewGelu.h"
#include "Kuiper_KB_NewGelu.cu"

static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_newgelu_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(),
                "kuiper_newgelu: expected a contiguous CUDA tensor");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 ||
                    X.scalar_type() == torch::kFloat64,
                "kuiper_newgelu: unsupported dtype");
    int64_t numel = X.numel();
    TORCH_CHECK(numel > 0 && numel <= KUIPER_MAX_NTHR,
                "kuiper_newgelu: numel outside the verified size domain");

    const c10::cuda::CUDAGuard device_guard(X.device());
    if (X.scalar_type() == torch::kFloat32) {
        float *output = Kuiper_KB_NewGelu_newgelu_alloc_f32(
            (uint32_t)numel, X.data_ptr<float>());
        return torch::from_blob(
            output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
    }
    double *output = Kuiper_KB_NewGelu_newgelu_alloc_f64(
        (uint32_t)numel, X.data_ptr<double>());
    return torch::from_blob(
        output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_newgelu", &kuiper_newgelu_cuda, "Kuiper verified MinGPT NewGELU");
}
