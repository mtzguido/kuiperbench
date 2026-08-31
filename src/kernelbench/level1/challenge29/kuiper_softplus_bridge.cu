// Bridge for KernelBench L1: Softplus.
// Applied in-place (computed on a clone) via verified
// Kuiper_KB_Softplus_softplus_fw_{f32,f64}.

#include <torch/extension.h>

#include "Kuiper_KB_Softplus.h"
#include "Kuiper_KB_Softplus.cu"

torch::Tensor kuiper_softplus_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "kuiper #29: expected a CUDA tensor");
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #29: element count exceeds the verified kernel bound");

    if (Y.scalar_type() == torch::kFloat32) {
        Kuiper_KB_Softplus_softplus_fw_f32(
            (uint32_t)numel, Y.data_ptr<float>());
    } else if (Y.scalar_type() == torch::kFloat64) {
        Kuiper_KB_Softplus_softplus_fw_f64(
            (uint32_t)numel, Y.data_ptr<double>());
    } else {
        TORCH_CHECK(false, "kuiper_softplus: unsupported dtype");
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_softplus", &kuiper_softplus_cuda, "Kuiper verified softplus");
}
