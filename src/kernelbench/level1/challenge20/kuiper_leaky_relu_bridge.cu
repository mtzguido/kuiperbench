// Bridge for KernelBench L1 #20: LeakyReLU.
// y[i] = x[i]                  if x[i] >= 0
//      = x[i] * negative_slope  otherwise
// Delegates to verified Kuiper_KB_LeakyReLU_leaky_relu_fw_f32.

#include <torch/extension.h>

#include "Kuiper_KB_LeakyReLU.h"
#include "Kuiper_KB_LeakyReLU.cu"

torch::Tensor kuiper_leaky_relu_cuda(torch::Tensor X, double negative_slope) {
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();

    Kuiper_KB_LeakyReLU_leaky_relu_fw_f32(
        (float)negative_slope, (uint32_t)numel, Y.data_ptr<float>());

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_leaky_relu_cuda", &kuiper_leaky_relu_cuda, "Kuiper verified LeakyReLU");
}
