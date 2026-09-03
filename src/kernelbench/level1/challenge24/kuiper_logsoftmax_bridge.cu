// Bridge for KernelBench L1 #24: LogSoftmax along dim=1.
// One self-allocating Kuiper call processes the entire (B, D) tensor.

#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_LogSoftmaxAlloc.h"
#include "Kuiper_KB_LogSoftmaxAlloc.cu"

torch::Tensor kuiper_log_softmax_cuda(torch::Tensor X) {
    TORCH_CHECK(X.dim() == 2, "kuiper_log_softmax: expected 2D input, got ",
                X.dim(), "D");
    TORCH_CHECK(X.is_cuda(), "kuiper_log_softmax: expected CUDA tensor");
    TORCH_CHECK(X.is_contiguous(),
                "kuiper_log_softmax: input must be contiguous");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 ||
                    X.scalar_type() == torch::kFloat64,
                "kuiper_log_softmax: unsupported dtype");
    TORCH_CHECK(X.size(0) > 0 && X.size(1) > 0,
                "kuiper_log_softmax: dimensions must be positive");
    TORCH_CHECK(X.size(0) <= UINT32_MAX && X.size(1) <= UINT32_MAX,
                "kuiper_log_softmax: dimensions exceed uint32 ABI");

    uint32_t B = (uint32_t)X.size(0);
    uint32_t D = (uint32_t)X.size(1);
    TORCH_CHECK(B <= 2097152,
                "kuiper_log_softmax: row count exceeds the verified launch bound");
    TORCH_CHECK((int64_t)B <= ((int64_t)2097152 * 1024) / D,
                "kuiper_log_softmax: element count exceeds the verified launch bound");

    const at::cuda::CUDAGuard device_guard(X.device());
    if (X.scalar_type() == torch::kFloat32) {
        float *output = Kuiper_KB_LogSoftmaxAlloc_logsoftmax_alloc_f32(
            B, D, X.data_ptr<float>());
        return torch::from_blob(
            output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
    }
    double *output = Kuiper_KB_LogSoftmaxAlloc_logsoftmax_alloc_f64(
        B, D, X.data_ptr<double>());
    return torch::from_blob(
        output, X.sizes(), [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_log_softmax", &kuiper_log_softmax_cuda,
          "Kuiper verified row-wise log_softmax along dim=1");
}
