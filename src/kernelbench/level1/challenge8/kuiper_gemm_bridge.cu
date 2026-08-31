// Bridge for KernelBench L1 #8: plain matmul C = A @ B via Naive3 (Kahan).

#include <torch/extension.h>
#include "Klas_GEMM_Naive3.h"
#include "Klas_GEMM_Naive3.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0),
                "kuiper #8: expected compatible 2-D tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "kuiper #8: expected float32 tensors");
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #8: tensors must be CUDA tensors on the same device");
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    int64_t rows = A_c.size(0);
    int64_t shared = A_c.size(1);
    int64_t cols = B_c.size(1);
    TORCH_CHECK(rows > 0 && shared > 0 && cols > 0 &&
                rows <= (int64_t)UINT32_MAX / shared &&
                shared <= (int64_t)UINT32_MAX / cols &&
                rows <= (int64_t)UINT32_MAX / cols &&
                rows <= ((int64_t)2097152 * 1024) / cols,
                "kuiper #8: shape exceeds the verified kernel bounds");
    auto C = torch::zeros({rows, cols}, A_c.options());
    Klas_GEMM_Naive3_g_matmul_f32_rrr(
        (uint32_t)rows, (uint32_t)cols, (uint32_t)shared,
        A_c.data_ptr<float>(), B_c.data_ptr<float>(), C.data_ptr<float>());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda, "Kuiper verified GEMM (Kahan)");
}
