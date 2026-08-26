// Bridge for KernelBench L1 #14: triu(A @ B). Plain GEMM + torch.triu glue.
#include <torch/extension.h>
#include "Klas_GEMM_Naive3.h"
#include "Klas_GEMM_Naive3.cu"

torch::Tensor kuiper_triu_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    int64_t rows = A_c.size(0), shared = A_c.size(1), cols = B_c.size(1);
    auto C = torch::zeros({rows, cols}, A_c.options());
    Klas_GEMM_Naive3_g_matmul_f32_rrr(
        (uint32_t)rows, (uint32_t)cols, (uint32_t)shared,
        A_c.data_ptr<float>(), B_c.data_ptr<float>(), C.data_ptr<float>());
    return torch::triu(C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_triu_matmul_cuda", &kuiper_triu_matmul_cuda, "Kuiper GEMM then triu");
}
