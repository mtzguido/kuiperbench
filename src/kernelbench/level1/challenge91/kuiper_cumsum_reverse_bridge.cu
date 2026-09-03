// Bridge for KernelBench L1 #91: reverse cumulative sum along dim=1.
//
// PyTorch: y = torch.cumsum(x.flip(1), dim=1).flip(1), shape (B, D), fp32.
//
// The reversal and scan composition are implemented by the verified Kuiper
// entry point. This bridge performs only ABI checks, selects the input device,
// makes one Kuiper call, and wraps the returned allocation.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_CumSumReverse.h"
#include "Kuiper_KB_CumSumReverse.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t) 2097152;

torch::Tensor kuiper_cumsum_reverse_dim1_cuda(torch::Tensor X)
{
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 &&
                    X.dim() == 2,
                "kuiper_cumsum_reverse_dim1: expected 2-D float32 CUDA tensor");
    TORCH_CHECK(X.is_contiguous(),
                "kuiper_cumsum_reverse_dim1: input must be contiguous");
    int64_t B = X.size(0), D = X.size(1);
    TORCH_CHECK(B > 0 && D > 0 && B <= (int64_t) UINT32_MAX &&
                    D <= (int64_t) UINT32_MAX &&
                    B <= (int64_t) UINT32_MAX / D && B <= KUIPER_MAX_BLOCKS,
                "kuiper_cumsum_reverse_dim1: shape out of range");
    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_CumSumReverse_cumsum_reverse_fw_f32(
        (uint32_t) B, (uint32_t) D, X.data_ptr<float>());
    return torch::from_blob(
        out, {B, D}, [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_cumsum_reverse_dim1", &kuiper_cumsum_reverse_dim1_cuda,
          "Kuiper verified reverse cumulative sum along dim=1 of a 2-D tensor");
}
