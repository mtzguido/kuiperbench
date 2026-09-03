// Bridge for KernelBench L1 #90: cumulative product along dim=1 of a
// (B, D) row-major float32 tensor.
//
// PyTorch: y = torch.cumprod(x, dim=1), shape (B, D), float32.
//
// Kuiper: same row-per-block sequential prefix-scan as cumsum, but
// parametrised by the law-free multiplicative reducer [reducer_fmul_f32].  One
// launch of [Kuiper_KB_CumProd_cumprod_fw_f32] does the whole forward.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_CumProd.h"
#include "Kuiper_KB_CumProd.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t)2097152;

torch::Tensor kuiper_cumprod_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 &&
                    X.dim() == 2 && X.is_contiguous(),
                "kuiper_cumprod_dim1: expected contiguous 2-D float32 CUDA tensor");
    int64_t B = X.size(0), D = X.size(1);
    TORCH_CHECK(B > 0 && D > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && (__int128) B * D <= UINT32_MAX
                && B <= KUIPER_MAX_BLOCKS,
                "kuiper_cumprod_dim1: shape out of range");
    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_CumProd_cumprod_alloc_f32(
        (uint32_t)B, (uint32_t)D, X.data_ptr<float>());
    return torch::from_blob(out, X.sizes(), [](void *p) { cudaFree(p); },
                            X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_cumprod_dim1", &kuiper_cumprod_dim1_cuda,
          "Kuiper verified cumulative product along dim=1 of a 2-D tensor");
}
