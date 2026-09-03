#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_RowScaleAlloc.h"
#include "Kuiper_KB_RowScaleAlloc.cu"

torch::Tensor kuiper_rowscale_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #12 inputs must be CUDA tensors on one device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                    B.scalar_type() == torch::kFloat32,
                "kuiper #12 inputs must be float32");
    TORCH_CHECK(A.dim() == 1 && B.dim() == 2 && A.size(0) == B.size(0),
                "kuiper #12 expected A=(M), B=(M,N)");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "kuiper #12 inputs must be contiguous");
    const int64_t m = A.size(0), n = B.size(1);
    TORCH_CHECK(m > 0 && n > 0 &&
                    (__int128) m * n <= (__int128) UINT32_MAX &&
                    (__int128) m * n <= (__int128) 2097152 * 1024,
                "kuiper #12 shape is outside the verified range");
    const c10::cuda::CUDAGuard device_guard(A.device());
    float *out = Kuiper_KB_RowScaleAlloc_row_scale_alloc_f32(
        (uint32_t) m, (uint32_t) n,
        A.data_ptr<float>(), B.data_ptr<float>());
    return torch::from_blob(
        out, {m, n}, [](void *p) { cudaFree(p); }, B.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_rowscale_cuda", &kuiper_rowscale_cuda,
          "Kuiper verified self-allocating diagonal matrix multiplication");
}
