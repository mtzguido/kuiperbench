// Bridge for KernelBench L1 #20: LeakyReLU.
// y[i] = x[i]                  if x[i] >= 0
//      = x[i] * negative_slope  otherwise
// Delegates to verified Kuiper_KB_LeakyReLU_leaky_relu_fw_f32.

#include <torch/extension.h>

#include "Kuiper_KB_LeakyReLU.h"
#include "Kuiper_KB_LeakyReLU.cu"

torch::Tensor kuiper_leaky_relu_cuda(torch::Tensor X, double negative_slope) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32,
                "kuiper #20: expected a float32 CUDA tensor");
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #20: element count exceeds the verified kernel bound");

    Kuiper_KB_LeakyReLU_leaky_relu_fw_f32(
        (float)negative_slope, (uint32_t)numel, Y.data_ptr<float>());

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_leaky_relu_cuda", &kuiper_leaky_relu_cuda, "Kuiper verified LeakyReLU");
}
