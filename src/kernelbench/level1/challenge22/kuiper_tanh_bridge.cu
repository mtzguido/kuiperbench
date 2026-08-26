// Bridge for KernelBench L1 #22: Tanh.
// y[i] = tanh(x[i]), applied in-place (computed on a clone) via
// verified Kuiper_KB_Tanh_tanh_fw_{f32,f64}.

#include <torch/extension.h>

#include "Kuiper_KB_Tanh.h"
#include "Kuiper_KB_Tanh.cu"

torch::Tensor kuiper_tanh_cuda(torch::Tensor X) {
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();

    if (Y.scalar_type() == torch::kFloat32) {
        Kuiper_KB_Tanh_tanh_fw_f32(
            (uint32_t)numel, Y.data_ptr<float>());
    } else if (Y.scalar_type() == torch::kFloat64) {
        Kuiper_KB_Tanh_tanh_fw_f64(
            (uint32_t)numel, Y.data_ptr<double>());
    } else {
        TORCH_CHECK(false, "kuiper_tanh: unsupported dtype");
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_tanh", &kuiper_tanh_cuda, "Kuiper verified tanh");
}
