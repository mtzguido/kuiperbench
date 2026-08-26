// Bridge for KernelBench L1 #11: 4D tensor @ matrix.
// A has shape (b, i, j, l); B has shape (l, k); result C has shape (b, i, j, k),
// with C[b,i,j,k] = sum_l A[b,i,j,l] * B[l,k]  (einsum "bijl,lk->bijk").
//
// Unlike a hand-rolled host-side flatten (numel/reshape + C.view), the
// (b,i,j)->(b*i*j) collapse is performed INSIDE the verified Kuiper kernel:
// Kuiper_KB_Matmul4D_matmul4d_f32 takes the raw row-major Array4 buffer, ghost-
// reshapes it to a row-major Array2 (b*i*j, l) over the SAME GPU buffer (no data
// movement), runs the verified Naive3 GEMM, and writes the (b,i,j,k) row-major
// result. The bridge only reads the five dimensions and passes pointers; it
// performs no reshape, transpose, or copy of its own.

#include <torch/extension.h>

#include "Kuiper_KB_Matmul4D.h"
#include "Kuiper_KB_Matmul4D.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 4, "A must be 4D (b,i,j,l), got ", A.dim(), "D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (l,k), got ", B.dim(), "D");
    TORCH_CHECK(A.size(3) == B.size(0),
                "shape mismatch: A is (", A.size(0), ",", A.size(1), ",", A.size(2),
                ",", A.size(3), "), B is (", B.size(0), ",", B.size(1), ")");
    // dtype is enforced by data_ptr<float>() below (throws on non-float).
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    int64_t b = A_c.size(0);
    int64_t i = A_c.size(1);
    int64_t j = A_c.size(2);
    int64_t l = A_c.size(3);
    int64_t k = B_c.size(1);

    TORCH_CHECK(b > 0 && i > 0 && j > 0 && l > 0 && k > 0, "dimensions must be positive");
    TORCH_CHECK(b <= UINT32_MAX && i <= UINT32_MAX && j <= UINT32_MAX
                && l <= UINT32_MAX && k <= UINT32_MAX,
                "dimensions exceed the uint32 ABI of the Kuiper kernel");

    // The product-overflow and launch-bound checks (b*i, b*i*j, b*i*j*l,
    // b*i*j*k, l*k all fit uint32, and (b*i*j)*k <= max_blocks*max_threads) are
    // now performed INSIDE the verified kernel via dguard (extracted KPR_GUARD),
    // so they need not be repeated here. Only the torch/ABI-boundary checks
    // above (which the kernel cannot see) remain.

    auto C = torch::empty({b, i, j, k}, A_c.options());

    Kuiper_KB_Matmul4D_matmul4d_f32(
        (uint32_t)b, (uint32_t)i, (uint32_t)j, (uint32_t)l, (uint32_t)k,
        A_c.data_ptr<float>(), B_c.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda,
          "Kuiper verified 4D tensor-matrix multiplication");
}
