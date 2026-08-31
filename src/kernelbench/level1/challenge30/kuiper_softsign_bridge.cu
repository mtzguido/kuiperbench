// Bridge for KernelBench L1: Softsign.
// Applied in-place (computed on a clone) via verified
// Kuiper_KB_Softsign_softsign_fw_{f32,f64}.

#include <torch/extension.h>

#include "Kuiper_KB_Softsign.h"
#include "Kuiper_KB_Softsign.cu"

torch::Tensor kuiper_softsign_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "kuiper #30: expected a CUDA tensor");
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #30: element count exceeds the verified kernel bound");

    if (Y.scalar_type() == torch::kFloat32) {
        Kuiper_KB_Softsign_softsign_fw_f32(
            (uint32_t)numel, Y.data_ptr<float>());
    } else if (Y.scalar_type() == torch::kFloat64) {
        Kuiper_KB_Softsign_softsign_fw_f64(
            (uint32_t)numel, Y.data_ptr<double>());
    } else {
        TORCH_CHECK(false, "kuiper_softsign: unsupported dtype");
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_softsign", &kuiper_softsign_cuda, "Kuiper verified softsign");
}
