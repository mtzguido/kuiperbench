// Bridge for KernelBench L1: HardTanh.
// Applied in-place (computed on a clone) via verified
// Kuiper_KB_HardTanh_htanh_fw_{f32,f64}.

#include <torch/extension.h>

#include "Kuiper_KB_HardTanh.h"
#include "Kuiper_KB_HardTanh.cu"

torch::Tensor kuiper_htanh_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "kuiper #32: expected a CUDA tensor");
    auto Y = X.contiguous().clone();
    int64_t numel = Y.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #32: element count exceeds the verified kernel bound");

    if (Y.scalar_type() == torch::kFloat32) {
        Kuiper_KB_HardTanh_htanh_fw_f32((uint32_t)numel, Y.data_ptr<float>());
    } else if (Y.scalar_type() == torch::kFloat64) {
        Kuiper_KB_HardTanh_htanh_fw_f64((uint32_t)numel, Y.data_ptr<double>());
    } else {
        TORCH_CHECK(false, "kuiper_htanh: unsupported dtype");
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_htanh", &kuiper_htanh_cuda, "Kuiper verified htanh");
}
