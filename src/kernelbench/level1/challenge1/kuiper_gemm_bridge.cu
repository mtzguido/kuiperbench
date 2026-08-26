// Bridge between extracted Kuiper GEMM and PyTorch.
// Uses verified BlockTiling2D GEMM (128x128x32, 8x8).
// Requires rows >= 128, cols >= 128, shared >= 32 (padding applied if needed).

#include <torch/extension.h>

#include "Klas_GEMM_Naive2.h"
#include "Klas_GEMM_Naive2.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D, got ", A.dim(), "D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D, got ", B.dim(), "D");
    TORCH_CHECK(A.size(1) == B.size(0),
                "shape mismatch: A is (", A.size(0), ",", A.size(1),
                "), B is (", B.size(0), ",", B.size(1), ")");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    int64_t rows = A_contig.size(0);
    int64_t shared = A_contig.size(1);
    int64_t cols = B_contig.size(1);
    TORCH_CHECK(rows > 0 && shared > 0 && cols > 0, "dimensions must be positive");
    TORCH_CHECK(rows <= UINT32_MAX && shared <= UINT32_MAX && cols <= UINT32_MAX,
                "dimensions exceed uint32 ABI of the Kuiper kernel");

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

    Klas_GEMM_Naive2_g_matmul_f32_rrr(
        (uint32_t)p_rows, (uint32_t)p_cols, (uint32_t)p_shared,
        gA.data_ptr<float>(), gB.data_ptr<float>(), gC.data_ptr<float>());

    // Slice back to original size
    return gC.slice(0, 0, rows).slice(1, 0, cols);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda, "Kuiper verified matmul");
}
