// Bridge for KernelBench L1 #23: Softmax along dim=1.
// Input shape (B, D). One verified Kuiper call processes all B rows.
// The extracted Klas_RowSoftmax_row_softmax_rm_f{32,64} runs a host-side
// for-loop over rows that calls the verified per-row sum reduction +
// per-row exp/divide kernels — the input tensor stays resident on the
// GPU end-to-end (one D2H of the per-row sum scalar; never the row data).

#include <torch/extension.h>

#include "Klas_RowSoftmax.h"
#include "Klas_RowSoftmax.cu"

torch::Tensor kuiper_softmax_cuda(torch::Tensor X) {
    TORCH_CHECK(X.dim() == 2, "kuiper_softmax: expected 2D input, got ",
                X.dim(), "D");
    TORCH_CHECK(X.is_cuda(), "kuiper_softmax: expected CUDA tensor");
    TORCH_CHECK(X.size(0) > 0 && X.size(1) > 0,
                "kuiper_softmax: dimensions must be positive");
    TORCH_CHECK(X.size(0) <= UINT32_MAX && X.size(1) <= UINT32_MAX,
                "kuiper_softmax: dimensions exceed uint32 ABI");

    auto Y = X.contiguous().clone();
    uint32_t B = (uint32_t)Y.size(0);
    uint32_t D = (uint32_t)Y.size(1);

    if (Y.scalar_type() == torch::kFloat32) {
        Klas_RowSoftmax_row_softmax_rm_f32(B, D, Y.data_ptr<float>());
    } else if (Y.scalar_type() == torch::kFloat64) {
        Klas_RowSoftmax_row_softmax_rm_f64(B, D, Y.data_ptr<double>());
    } else {
        TORCH_CHECK(false, "kuiper_softmax: unsupported dtype");
    }

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_softmax", &kuiper_softmax_cuda,
          "Kuiper verified row-wise softmax along dim=1");
}
