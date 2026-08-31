// Bridge for KernelBench L1 #4: Matrix-vector multiplication.
// A (M, K) × B (K, 1) -> C (M, 1). Use Naive2 GEMM directly (N=1).
// BlockTiling2D requires cols >= 128, but N=1 here, so Naive2 is the right fit.

#include <torch/extension.h>

// #include "Klas_GEMM_Naive2.h"
// #include "Klas_GEMM_Naive2.cu"
#include "Klas_GEMM_Naive3.h"
#include "Klas_GEMM_Naive3.cu"

torch::Tensor kuiper_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && B.size(1) == 1,
                "kuiper #4: expected A=(M,K), B=(K,1)");
    TORCH_CHECK(A.size(1) == B.size(0), "kuiper #4: shape mismatch");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                B.scalar_type() == torch::kFloat32,
                "kuiper #4: expected float32 tensors");
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #4: tensors must be CUDA tensors on the same device");
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    int64_t rows = A_contig.size(0);
    int64_t shared = A_contig.size(1);
    int64_t cols = B_contig.size(1);
    TORCH_CHECK(rows > 0 && shared > 0 &&
                rows <= (int64_t)UINT32_MAX &&
                shared <= (int64_t)UINT32_MAX &&
                rows <= (int64_t)UINT32_MAX / shared &&
                shared <= (int64_t)UINT32_MAX / cols &&
                rows <= ((int64_t)2097152 * 1024) / cols,
                "kuiper #4: shape exceeds the verified kernel bounds");

    auto gC = torch::zeros({rows, cols}, A_contig.options());

    Klas_GEMM_Naive3_g_matmul_f32_rrr(
        (uint32_t)rows, (uint32_t)cols, (uint32_t)shared,
        A_contig.data_ptr<float>(), B_contig.data_ptr<float>(), gC.data_ptr<float>());

    return gC;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matvec_cuda", &kuiper_matvec_cuda, "Kuiper verified matrix-vector mul");
}
