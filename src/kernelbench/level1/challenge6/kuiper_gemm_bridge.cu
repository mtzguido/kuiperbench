// Bridge for KernelBench L1 #6: Matmul with large K (M=256, N=256, K=2^19).
// Kahan-accumulated Naive3 GEMM: avoids fp32 rounding drift over K=524288.

#include <torch/extension.h>

#include "Klas_GEMM_Naive1.h"
#include "Klas_GEMM_Naive1.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    int64_t rows = A_contig.size(0);
    int64_t shared = A_contig.size(1);
    int64_t cols = B_contig.size(1);

    auto gC = torch::zeros({rows, cols}, A_contig.options());

    Klas_GEMM_Naive1_g_matmul_f32_rrr(
        (uint32_t)rows, (uint32_t)cols, (uint32_t)shared,
        A_contig.data_ptr<float>(), B_contig.data_ptr<float>(), gC.data_ptr<float>());

    return gC;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda, "Kuiper verified GEMM (Kahan)");
}
