// Bridge for KernelBench L1 #12: Matmul with diagonal matrix.
// C = diag(A) @ B, equivalently C[i,j] = A[i] * B[i,j].
// A is (M,), B is (M, N); we treat B as a flat M*N row-major array.
// Implemented in-place on a clone of B using verified
// Kuiper_KB_RowScale_row_scale_f32 (1024-thread blocks, bounds-checked).

#include <torch/extension.h>

#include "Klas_RowScale.h"
#include "Klas_RowScale.cu"

torch::Tensor kuiper_rowscale_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 1 && B.dim() == 2 && A.size(0) == B.size(0),
                "kuiper #12: expected A=(M), B=(M,N)");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "kuiper #12: expected float32 tensors");
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #12: tensors must be CUDA tensors on the same device");
    auto A_c = A.contiguous();
    auto B_c = B.contiguous().clone();  // in-place kernel mutates B_c
    int64_t m = A_c.size(0);
    int64_t n = B_c.size(1);
    TORCH_CHECK(m > 0 && n > 0 &&
                m <= (int64_t)UINT32_MAX / n &&
                m <= ((int64_t)2097152 * 1024) / n,
                "kuiper #12: shape exceeds the verified kernel bounds");
    uint32_t M = (uint32_t)m;
    uint32_t N = (uint32_t)n;

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
