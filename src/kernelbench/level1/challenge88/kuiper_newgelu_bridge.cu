// Bridge for KernelBench L1 #88: MinGPT NewGELU.
//   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Applied in-place (computed on a clone) via verified
// Kuiper_KB_NewGelu_newgelu_fw_{f32,f64}.

#include <torch/extension.h>

#include "Kuiper_KB_NewGelu.h"
#include "Kuiper_KB_NewGelu.cu"

torch::Tensor kuiper_newgelu_cuda(torch::Tensor X) {
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();

    // sqrt(2/pi) to ~20 digits.
    if (Y.scalar_type() == torch::kFloat32) {
        Kuiper_KB_NewGelu_newgelu_fw_f32(
            0.5f, 0.79788456080286535588f, 0.044715f,
            (uint32_t)numel, Y.data_ptr<float>());
    } else if (Y.scalar_type() == torch::kFloat64) {
        Kuiper_KB_NewGelu_newgelu_fw_f64(
            0.5, 0.79788456080286535588, 0.044715,
            (uint32_t)numel, Y.data_ptr<double>());
    } else {
        TORCH_CHECK(false, "kuiper_newgelu: unsupported dtype");
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_newgelu", &kuiper_newgelu_cuda, "Kuiper verified MinGPT NewGELU");
}
