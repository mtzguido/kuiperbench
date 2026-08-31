// Bridge for KernelBench L1 #24: LogSoftmax along dim=1.
// One verified Kuiper call processes the entire (B, D) tensor in two
// GPU kernel launches via Klas_RowLogSoftmax (which composes a
// block-per-row tree reduction of exp(x) with a per-cell `x - log(sum)`
// row-broadcast map).

#include <torch/extension.h>

#include "Klas_RowLogSoftmax.h"
#include "Klas_RowLogSoftmax.cu"

torch::Tensor kuiper_log_softmax_cuda(torch::Tensor X) {
    TORCH_CHECK(X.dim() == 2, "kuiper_log_softmax: expected 2D input, got ",
                X.dim(), "D");
    TORCH_CHECK(X.is_cuda(), "kuiper_log_softmax: expected CUDA tensor");
    TORCH_CHECK(X.size(0) > 0 && X.size(1) > 0,
                "kuiper_log_softmax: dimensions must be positive");
    TORCH_CHECK(X.size(0) <= UINT32_MAX && X.size(1) <= UINT32_MAX,
                "kuiper_log_softmax: dimensions exceed uint32 ABI");

    auto Y = X.contiguous().clone();
    uint32_t B = (uint32_t)Y.size(0);
    uint32_t D = (uint32_t)Y.size(1);
    TORCH_CHECK(B <= 2097152,
                "kuiper_log_softmax: row count exceeds the verified launch bound");
    TORCH_CHECK((int64_t)B <= ((int64_t)2097152 * 1024) / D,
                "kuiper_log_softmax: element count exceeds the verified launch bound");

    if (Y.scalar_type() == torch::kFloat32) {
        Klas_RowLogSoftmax_row_log_softmax_rm_f32(B, D, Y.data_ptr<float>());
    } else if (Y.scalar_type() == torch::kFloat64) {
        Klas_RowLogSoftmax_row_log_softmax_rm_f64(B, D, Y.data_ptr<double>());
    } else {
        TORCH_CHECK(false, "kuiper_log_softmax: unsupported dtype");
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_log_softmax", &kuiper_log_softmax_cuda,
          "Kuiper verified row-wise log_softmax along dim=1");
}
