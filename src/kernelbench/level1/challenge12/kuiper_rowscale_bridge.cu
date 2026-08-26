// Bridge for KernelBench L1 #12: Matmul with diagonal matrix.
// C = diag(A) @ B, equivalently C[i,j] = A[i] * B[i,j].
// A is (M,), B is (M, N); we treat B as a flat M*N row-major array.
// Implemented in-place on a clone of B using verified
// Kuiper_KB_RowScale_row_scale_f32 (1024-thread blocks, bounds-checked).

#include <torch/extension.h>

#include "Klas_RowScale.h"
#include "Klas_RowScale.cu"

torch::Tensor kuiper_rowscale_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_c = A.contiguous();
    auto B_c = B.contiguous().clone();  // in-place kernel mutates B_c
    uint32_t M = (uint32_t)A_c.size(0);
    uint32_t N = (uint32_t)B_c.size(1);

    Klas_RowScale_rowscale_f32_rowmajor(
        M, N,
        A_c.data_ptr<float>(),
        B_c.data_ptr<float>());

    return B_c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_rowscale_cuda", &kuiper_rowscale_cuda,
          "Kuiper verified diagonal matrix multiplication (diag(A) @ B)");
}
