// Bridge between extracted Kuiper GEMM and PyTorch.
// Uses BlockTiling2D GEMM (128x128x32, 8x8) — the fastest verified variant.
// Requires rows >= 128, cols >= 128, shared >= 32 (padding applied if needed).

#include <torch/extension.h>

#include "Klas_GEMM_Naive3.h"
#include "Klas_GEMM_Naive3.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    int64_t rows = A_contig.size(0);
    int64_t shared = A_contig.size(1);
    int64_t cols = B_contig.size(1);

    // Pad to minimum tile sizes if needed
    int64_t p_rows = (rows + 127) / 128 * 128;
    int64_t p_shared = (shared + 31) / 32 * 32;
    int64_t p_cols = (cols + 127) / 128 * 128;

    auto gA = (p_rows == rows && p_shared == shared)
        ? A_contig
        : torch::nn::functional::pad(A_contig, torch::nn::functional::PadFuncOptions({0, (int)(p_shared - shared), 0, (int)(p_rows - rows)}));
    auto gB = (p_shared == shared && p_cols == cols)
        ? B_contig
        : torch::nn::functional::pad(B_contig, torch::nn::functional::PadFuncOptions({0, (int)(p_cols - cols), 0, (int)(p_shared - shared)}));
    auto gC = torch::zeros({p_rows, p_cols}, A_contig.options());

    Klas_GEMM_Naive3_g_matmul_f32_rrr(
        (uint32_t)p_rows, (uint32_t)p_cols, (uint32_t)p_shared,
        gA.data_ptr<float>(), gB.data_ptr<float>(), gC.data_ptr<float>());

    // Slice back to original size
    return gC.slice(0, 0, rows).slice(1, 0, cols);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda, "Kuiper verified matmul");
}
