#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_MatmulND.h"
#include "Kuiper_KB_MatmulND.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #10 inputs must be CUDA tensors on one device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                    B.scalar_type() == torch::kFloat32,
                "kuiper #10 inputs must be float32");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 2 && A.size(2) == B.size(0),
                "kuiper #10 expected A=(N,M,K), B=(K,L)");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "kuiper #10 inputs must be contiguous");
    const int64_t n = A.size(0), m = A.size(1), k = A.size(2), l = B.size(1);
    const __int128 umax = (__int128) UINT32_MAX;
    const __int128 launch_max = (__int128) 2097152 * 1024;
    TORCH_CHECK(n > 0 && m > 0 && k > 0 && l > 0 &&
                    (__int128) n * m * k <= umax &&
                    (__int128) k * l <= umax &&
                    (__int128) n * m * l <= umax &&
                    (__int128) n * m * l <= launch_max,
                "kuiper #10 shape is outside the verified range");
    const c10::cuda::CUDAGuard device_guard(A.device());
    float *out = Kuiper_KB_MatmulND_matmul_nd_alloc_f32(
        (uint32_t) n, (uint32_t) m, (uint32_t) k, (uint32_t) l,
        A.data_ptr<float>(), B.data_ptr<float>());
    return torch::from_blob(
        out, {n, m, l}, [](void *p) { cudaFree(p); }, A.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda,
          "Kuiper verified self-allocating 3D tensor-matrix multiplication");
}
