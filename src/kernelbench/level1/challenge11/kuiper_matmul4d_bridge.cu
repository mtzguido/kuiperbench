#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_Matmul4D.h"
#include "Kuiper_KB_Matmul4D.cu"

torch::Tensor kuiper_matmul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #11 inputs must be CUDA tensors on one device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                    B.scalar_type() == torch::kFloat32,
                "kuiper #11 inputs must be float32");
    TORCH_CHECK(A.dim() == 4 && B.dim() == 2 && A.size(3) == B.size(0),
                "kuiper #11 expected A=(b,i,j,l), B=(l,k)");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "kuiper #11 inputs must be contiguous");
    const int64_t b = A.size(0), i = A.size(1), j = A.size(2);
    const int64_t l = A.size(3), k = B.size(1);
    const __int128 umax = (__int128) UINT32_MAX;
    const __int128 launch_max = (__int128) 2097152 * 1024;
    TORCH_CHECK(b > 0 && i > 0 && j > 0 && l > 0 && k > 0 &&
                    (__int128) b * i * j * l <= umax &&
                    (__int128) l * k <= umax &&
                    (__int128) b * i * j * k <= umax &&
                    (__int128) b * i * j * k <= launch_max,
                "kuiper #11 shape is outside the verified range");
    const c10::cuda::CUDAGuard device_guard(A.device());
    float *out = Kuiper_KB_Matmul4D_matmul4d_alloc_f32(
        (uint32_t) b, (uint32_t) i, (uint32_t) j, (uint32_t) l, (uint32_t) k,
        A.data_ptr<float>(), B.data_ptr<float>());
    return torch::from_blob(
        out, {b, i, j, k}, [](void *p) { cudaFree(p); }, A.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_matmul_cuda", &kuiper_matmul_cuda,
          "Kuiper verified self-allocating 4D tensor-matrix multiplication");
}
