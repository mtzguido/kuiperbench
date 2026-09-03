// Bridge for KernelBench L1 #20: LeakyReLU.
// y[i] = x[i]                  if x[i] >= 0
//      = x[i] * negative_slope  otherwise
// Kuiper allocates and computes the result from the original input.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_LeakyReLU.h"
#include "Kuiper_KB_LeakyReLU.cu"

torch::Tensor kuiper_leaky_relu_cuda(torch::Tensor X, double negative_slope) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32,
                "kuiper #20: expected a float32 CUDA tensor");
    TORCH_CHECK(X.is_contiguous(), "kuiper #20: input must be contiguous");
    int64_t numel = X.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #20: element count exceeds the verified kernel bound");

    const c10::cuda::CUDAGuard device_guard(X.device());
    float *output = Kuiper_KB_LeakyReLU_leaky_relu_alloc_f64_f32(
        negative_slope, (uint32_t)numel, X.data_ptr<float>());
    return torch::from_blob(
        output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_leaky_relu_cuda", &kuiper_leaky_relu_cuda, "Kuiper verified LeakyReLU");
}
