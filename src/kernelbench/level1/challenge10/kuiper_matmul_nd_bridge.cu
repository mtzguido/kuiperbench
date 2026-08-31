// Bridge for KernelBench L1 #10: 3D tensor @ matrix.
// A has shape (N, M, K); B has shape (K, L); result C has shape (N, M, L),
// with C[n][m][:] = A[n][m][:] @ B (same matrix B for every (n,m) slice).
//
// Unlike a hand-rolled flatten, the (N,M)->(N*M) collapse is performed INSIDE
// the verified Kuiper kernel: Kuiper_KB_MatmulND.matmul_nd_f32 takes the raw
// row-major Array3 buffer, ghost-reshapes it to a row-major Array2 (N*M, K)
// over the SAME GPU buffer (no data movement), runs the verified Naive3 GEMM,
// and writes the (N,M,L) row-major result. The bridge only passes pointers and
// dimensions; it performs no reshape, transpose, or copy of its own.

#include <torch/extension.h>

#include "Kuiper_KB_MatmulND.h"
#include "Kuiper_KB_MatmulND.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D (N,M,K), got ", A.dim(), "D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K,L), got ", B.dim(), "D");
    TORCH_CHECK(A.size(2) == B.size(0),
                "shape mismatch: A is (", A.size(0), ",", A.size(1), ",", A.size(2),
                "), B is (", B.size(0), ",", B.size(1), ")");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    int64_t n = A_c.size(0);
    int64_t m = A_c.size(1);
    int64_t k = A_c.size(2);
    int64_t l = B_c.size(1);

    TORCH_CHECK(n > 0 && m > 0 && k > 0 && l > 0, "dimensions must be positive");
    TORCH_CHECK(n <= UINT32_MAX && m <= UINT32_MAX && k <= UINT32_MAX && l <= UINT32_MAX,
                "dimensions exceed the uint32 ABI of the Kuiper kernel");
    // Every GPU array length must fit the kernel's uint32 ABI (SZ.fits <=> u32):
    //   gA has length N*M*K, gB has length K*L, gC has length N*M*L.
    // Per-dim checks above don't bound these products, so guard them explicitly;
    // otherwise a uint32 index multiply could wrap and read out of bounds. We use
    // division so the guard arithmetic itself can't overflow int64.
    TORCH_CHECK(n <= (int64_t)UINT32_MAX / m,
                "N*M exceeds the uint32 length bound");
    int64_t nm = n * m;
    TORCH_CHECK(nm <= (int64_t)UINT32_MAX / k,
                "N*M*K exceeds the uint32 length bound of the Kuiper kernel");
    TORCH_CHECK(k <= (int64_t)UINT32_MAX / l,
                "K*L exceeds the uint32 length bound of the Kuiper kernel");
    // The kernel's size_req is (n*m)*l <= max_blocks * max_threads (~2.1e9).
    TORCH_CHECK(nm <= ((int64_t)2097152 * 1024) / l,
                "N*M*L exceeds the kernel's launch bound");

    auto C = torch::empty({n, m, l}, A_c.options());

    Kuiper_KB_MatmulND_matmul_nd_f32(
        (uint32_t)n, (uint32_t)m, (uint32_t)k, (uint32_t)l,
        A_c.data_ptr<float>(), B_c.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda,
          "Kuiper verified 3D tensor-matrix multiplication");
}
