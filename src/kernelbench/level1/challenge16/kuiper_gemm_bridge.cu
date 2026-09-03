#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_TransposedGEMM.h"
#include "Kuiper_KB_TransposedGEMM.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #16 inputs must be CUDA tensors on one device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                    B.scalar_type() == torch::kFloat32,
                "kuiper #16 inputs must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(0) == B.size(0),
                "kuiper #16 expected A=(K,M), B=(K,N)");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "kuiper #16 inputs must be contiguous");
    const int64_t k = A.size(0), m = A.size(1), n = B.size(1);
    const __int128 umax = (__int128) UINT32_MAX;
    TORCH_CHECK(m > 0 && n > 0 && k > 0 &&
                    (__int128) k * m <= umax &&
                    (__int128) k * n <= umax &&
                    (__int128) m * n <= umax &&
                    (__int128) m * n <= (__int128) 2097152 * 1024,
                "kuiper #16 shape is outside the verified range");
    const c10::cuda::CUDAGuard device_guard(A.device());
    float *out = Kuiper_KB_TransposedGEMM_matmul_f32_atb_alloc(
        (uint32_t) m, (uint32_t) n, (uint32_t) k,
        A.data_ptr<float>(), B.data_ptr<float>());
    return torch::from_blob(
        out, {m, n}, [](void *p) { cudaFree(p); }, A.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda,
          "Kuiper verified self-allocating A.T @ B");
}
