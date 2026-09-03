#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_TriuMatmul.h"
#include "Kuiper_KB_TriuMatmul.cu"

torch::Tensor kuiper_triu_matmul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #14 inputs must be CUDA tensors on one device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                    B.scalar_type() == torch::kFloat32,
                "kuiper #14 inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(0) == A.size(1) &&
                    B.sizes() == A.sizes(),
                "kuiper #14 expected equal square matrices");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "kuiper #14 inputs must be contiguous");
    const int64_t n = A.size(0);
    TORCH_CHECK(n > 0 && (__int128) n * n <= (__int128) UINT32_MAX &&
                    (__int128) n * n <= (__int128) 2097152 * 1024,
                "kuiper #14 shape is outside the verified range");
    const at::cuda::CUDAGuard device_guard(A.device());
    float *out = Kuiper_KB_TriuMatmul_triu_matmul_alloc_f32(
        (uint32_t) n, A.data_ptr<float>(), B.data_ptr<float>());
    return torch::from_blob(
        out, {n, n}, [](void *p) { cudaFree(p); }, A.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_triu_matmul_cuda", &kuiper_triu_matmul_cuda,
          "Kuiper verified self-allocating upper-triangular matmul");
}
