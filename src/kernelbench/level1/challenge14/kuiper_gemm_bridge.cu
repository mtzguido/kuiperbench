// Bridge for KernelBench L1 #14: verified triu(A @ B).
#include <torch/extension.h>
#include "Kuiper_KB_TriuMatmul.h"
#include "Kuiper_KB_TriuMatmul.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_triu_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kFloat32 && A.dim() == 2,
                "kuiper_triu_matmul: expected 2-D float32 CUDA tensor A");
    TORCH_CHECK(B.is_cuda() && B.scalar_type() == torch::kFloat32 && B.dim() == 2,
                "kuiper_triu_matmul: expected 2-D float32 CUDA tensor B");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

    auto Ac = A.contiguous();
    auto Bc = B.contiguous();
    int64_t n = Ac.size(0);
    TORCH_CHECK(Ac.size(1) == n && Bc.size(0) == n && Bc.size(1) == n,
                "kuiper_triu_matmul: A and B must be square and equal-shaped (N, N)");
    TORCH_CHECK(n > 0 && n <= (int64_t)UINT32_MAX / n
                    && n <= (KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS) / n,
                "kuiper_triu_matmul: shape out of range for the verified kernel ABI");

    auto Y = torch::empty({n * n}, Ac.options());
    Kuiper_KB_TriuMatmul_triu_matmul_f32(
        (uint32_t)n, Ac.data_ptr<float>(), Bc.data_ptr<float>(), Y.data_ptr<float>());

    return Y.reshape({n, n});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_triu_matmul_cuda", &kuiper_triu_matmul_cuda,
          "Kuiper verified upper-triangular matmul");
}
