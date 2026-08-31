// Bridge between extracted Kuiper GEMM and PyTorch.
// Uses the verified Naive3 GEMM.  Naive2's exported rank-2 contract limits
// rows*cols to max_blocks, which excludes KernelBench's 4096x4096 output.

#include <torch/extension.h>

#include "Klas_GEMM_Naive3.h"
#include "Klas_GEMM_Naive3.cu"

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
    TORCH_CHECK(rows <= (int64_t)UINT32_MAX &&
                shared <= (int64_t)UINT32_MAX &&
                cols <= (int64_t)UINT32_MAX,
                "dimensions exceed uint32 ABI of the Kuiper kernel");
    TORCH_CHECK(rows <= (int64_t)UINT32_MAX / shared &&
                shared <= (int64_t)UINT32_MAX / cols &&
                rows <= (int64_t)UINT32_MAX / cols &&
                rows <= ((int64_t)2097152 * 1024) / cols,
                "shape exceeds the verified kernel bounds");

    auto gC = torch::zeros({rows, cols}, A_contig.options());

    Klas_GEMM_Naive3_g_matmul_f32_rrr(
        (uint32_t)rows, (uint32_t)cols, (uint32_t)shared,
        A_contig.data_ptr<float>(), B_contig.data_ptr<float>(),
        gC.data_ptr<float>());

    return gC;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda, "Kuiper verified matmul");
}
