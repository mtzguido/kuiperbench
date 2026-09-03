#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_ScalarMul.h"
#include "Kuiper_KB_ScalarMul.cu"

torch::Tensor kuiper_smul_cuda(torch::Tensor A, double s)
{
    TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kFloat32,
                "kuiper #5 expected a float32 CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "kuiper #5 input must be contiguous");
    const int64_t numel = A.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t) 2097152 * 1024,
                "kuiper #5 element count exceeds the verified range");
    const c10::cuda::CUDAGuard device_guard(A.device());
    float *out = Kuiper_KB_ScalarMul_smul_alloc_f64_f32(
        s, (uint32_t) numel, A.data_ptr<float>());
    return torch::from_blob(
        out, A.sizes(), [](void *p) { cudaFree(p); }, A.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_smul_cuda", &kuiper_smul_cuda,
          "Kuiper verified self-allocating matrix-scalar multiplication");
}
