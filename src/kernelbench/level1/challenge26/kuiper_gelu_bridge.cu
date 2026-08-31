// Bridge for KernelBench L1: Gelu.
// Applied in-place (computed on a clone) via verified
// Kuiper_KB_Gelu_gelu_fw_{f32,f64}.

#include <torch/extension.h>

#include "Kuiper_KB_Gelu.h"
#include "Kuiper_KB_Gelu.cu"

torch::Tensor kuiper_gelu_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "kuiper #26: expected a CUDA tensor");
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #26: element count exceeds the verified kernel bound");

    if (Y.scalar_type() == torch::kFloat32) {
        Kuiper_KB_Gelu_gelu_fw_f32((uint32_t)numel, Y.data_ptr<float>());
    } else if (Y.scalar_type() == torch::kFloat64) {
        Kuiper_KB_Gelu_gelu_fw_f64((uint32_t)numel, Y.data_ptr<double>());
    } else {
        TORCH_CHECK(false, "kuiper_gelu: unsupported dtype");
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_gelu", &kuiper_gelu_cuda, "Kuiper verified gelu");
}
