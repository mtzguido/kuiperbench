#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_BatchedGEMM.h"
#include "Kuiper_KB_BatchedGEMM.cu"

torch::Tensor kuiper_batched_gemm_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device(),
                "kuiper #3 inputs must be CUDA tensors on one device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 &&
                    B.scalar_type() == torch::kFloat32,
                "kuiper #3 inputs must be float32");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 &&
                    A.size(0) == B.size(0) && A.size(2) == B.size(1),
                "kuiper #3 expected compatible batched matrices");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "kuiper #3 inputs must be contiguous");
    const int64_t batch = A.size(0), rows = A.size(1);
    const int64_t shared = A.size(2), cols = B.size(2);
    const __int128 umax = (__int128) UINT32_MAX;
    const __int128 launch_max = (__int128) 2097152 * 1024;
    TORCH_CHECK(batch > 0 && rows > 0 && shared > 0 && cols > 0 &&
                    (__int128) batch * rows * shared <= umax &&
                    (__int128) batch * shared * cols <= umax &&
                    (__int128) batch * rows * cols <= umax &&
                    (__int128) batch * rows * cols <= launch_max,
                "kuiper #3 shape is outside the verified range");
    const c10::cuda::CUDAGuard device_guard(A.device());
    float *out = Kuiper_KB_BatchedGEMM_batched_gemm_alloc_f32(
        (uint32_t) batch, (uint32_t) rows, (uint32_t) shared, (uint32_t) cols,
        A.data_ptr<float>(), B.data_ptr<float>());
    return torch::from_blob(
        out, {batch, rows, cols}, [](void *p) { cudaFree(p); }, A.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_batched_gemm_cuda", &kuiper_batched_gemm_cuda,
          "Kuiper verified self-allocating batched matrix multiplication");
}
