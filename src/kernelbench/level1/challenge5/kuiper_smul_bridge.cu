// Bridge for KernelBench L1 #5: Matrix-scalar multiplication.
// C = A * s   where A is (M, N), s is a scalar.
// Uses verified Kuiper_KB_ScalarMul.smul_out_f32 (out-of-place, 1024-thread
// blocks, bounds-checked): reads A, writes a fresh output buffer, leaving A
// untouched.  No clone of the input is needed.

#include <torch/extension.h>

#include "Kuiper_KB_ScalarMul.h"
#include "Kuiper_KB_ScalarMul.cu"

torch::Tensor kuiper_smul_cuda(torch::Tensor A, double s) {
    TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kFloat32,
                "kuiper #5: expected a float32 CUDA tensor");
    auto A_contig = A.contiguous();
    auto C = torch::empty_like(A_contig);
    int64_t numel = A_contig.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)2097152 * 1024,
                "kuiper #5: element count exceeds the verified kernel bound");

    Kuiper_KB_ScalarMul_smul_out_f32(
        (float)s, (uint32_t)numel,
        C.data_ptr<float>(), A_contig.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_smul_cuda", &kuiper_smul_cuda, "Kuiper verified matrix-scalar multiplication");
}
