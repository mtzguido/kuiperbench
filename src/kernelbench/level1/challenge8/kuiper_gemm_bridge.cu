#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_GEMMAlloc.h"
#include "Kuiper_KB_GEMMAlloc.cu"

static torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "Kuiper GEMM inputs must be CUDA tensors on one device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                    B.scalar_type() == torch::kFloat32,
                "Kuiper GEMM inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0),
                "Kuiper GEMM expected compatible rank-2 inputs");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "Kuiper GEMM inputs must be contiguous");

    const int64_t m = A.size(0), k = A.size(1), n = B.size(1);
    TORCH_CHECK(m > 0 && n > 0 && k > 0 && m <= (int64_t) UINT32_MAX &&
                    n <= (int64_t) UINT32_MAX && k <= (int64_t) UINT32_MAX,
                "Kuiper GEMM dimensions exceed the uint32 ABI");

    const c10::cuda::CUDAGuard device_guard(A.device());
    float *out = Kuiper_KB_GEMMAlloc_gemm_naive3_alloc_f32(
        (uint32_t) m, (uint32_t) n, (uint32_t) k, A.data_ptr<float>(),
        B.data_ptr<float>());
    return torch::from_blob(
        out, {m, n}, [](void *p) { cudaFree(p); }, A.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda,
          "Kuiper verified self-allocating matmul");
}
