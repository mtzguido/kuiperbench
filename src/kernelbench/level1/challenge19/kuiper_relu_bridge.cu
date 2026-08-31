// Bridge for KernelBench L1 #19: ReLU.
// y[i] = max(x[i], 0), applied in-place (computed on a clone) via verified
// Kuiper_KB_LeakyReLU_leaky_relu_fw_f32 with slope = 0.0f.

#include <torch/extension.h>

#include "Kuiper_KB_LeakyReLU.h"
#include "Kuiper_KB_LeakyReLU.cu"

torch::Tensor kuiper_relu_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32,
                "kuiper #19: expected a float32 CUDA tensor");
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #19: element count exceeds the verified kernel bound");

    Kuiper_KB_LeakyReLU_leaky_relu_fw_f32(
        0.0f, (uint32_t)numel, Y.data_ptr<float>());

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_relu_cuda", &kuiper_relu_cuda, "Kuiper verified ReLU");
}
