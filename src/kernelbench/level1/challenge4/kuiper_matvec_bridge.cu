// Bridge for KernelBench L1 #4: Matrix-vector multiplication.
// A (M, K) × B (K, 1) -> C (M, 1). Use Naive2 GEMM directly (N=1).
// BlockTiling2D requires cols >= 128, but N=1 here, so Naive2 is the right fit.

#include <torch/extension.h>

// #include "Klas_GEMM_Naive2.h"
// #include "Klas_GEMM_Naive2.cu"
#include "Klas_GEMM_Naive3.h"
#include "Klas_GEMM_Naive3.cu"

torch::Tensor kuiper_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    int64_t rows = A_contig.size(0);
    int64_t shared = A_contig.size(1);
    int64_t cols = B_contig.size(1);

    auto gC = torch::zeros({rows, cols}, A_contig.options());

    Klas_GEMM_Naive3_g_matmul_f32_rrr(
        (uint32_t)rows, (uint32_t)cols, (uint32_t)shared,
        A_contig.data_ptr<float>(), B_contig.data_ptr<float>(), gC.data_ptr<float>());

    return gC;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matvec_cuda", &kuiper_matvec_cuda, "Kuiper verified matrix-vector mul");
}
