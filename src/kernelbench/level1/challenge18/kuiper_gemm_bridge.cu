#include <torch/extension.h>
#include "Kuiper_KB_TransposedGEMM.h"
#include "Kuiper_KB_TransposedGEMM.cu"

// KernelBench L1 #18: C = A.T @ B.T,  A:(K,M), B:(N,K) -> C:(M,N).
//
// The transpose is captured INSIDE the verification boundary: all operands are
// passed ROW-MAJOR and the verified kernel ghost-transposes BOTH A and B
// internally (Kuiper.Ghost.TensorTranspose) before the GEMM, with a
// postcondition stating `C ~ matmul (mtranspose A) (mtranspose B)` explicitly.
// No physical transpose, no bridge-level layout reinterpretation.
torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kFloat32 && A.dim() == 2,
                "kuiper #18: expected 2-D float32 CUDA tensor A");
    TORCH_CHECK(B.is_cuda() && B.scalar_type() == torch::kFloat32 && B.dim() == 2,
                "kuiper #18: expected 2-D float32 CUDA tensor B");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

    auto A_c = A.contiguous();                 // (K, M) row-major
    auto B_c = B.contiguous();                 // (N, K) row-major
    int64_t K = A_c.size(0), M = A_c.size(1), N = B_c.size(0);
    TORCH_CHECK(B_c.size(1) == K,
                "shape mismatch: A is (", K, ",", M, "), B is (", N, ",", B_c.size(1), ")");
    TORCH_CHECK(M > 0 && N > 0 && K > 0
                && M <= (int64_t)UINT32_MAX && N <= (int64_t)UINT32_MAX
                && K <= (int64_t)UINT32_MAX
                && M * K <= (int64_t)UINT32_MAX && K * N <= (int64_t)UINT32_MAX
                && M * N <= (int64_t)2097152 * 1024,
                "kuiper #18: shape out of range for the verified kernel ABI");

    auto C = torch::zeros({M, N}, A_c.options());
    Kuiper_KB_TransposedGEMM_matmul_f32_atbt(
        (uint32_t)M, (uint32_t)N, (uint32_t)K,
        A_c.data_ptr<float>(), B_c.data_ptr<float>(), C.data_ptr<float>());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda, "Kuiper verified GEMM A.T @ B.T (no transpose)");
}
