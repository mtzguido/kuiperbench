// Bridge for KernelBench L1 #93: masked cumulative sum along dim=1.
//
// PyTorch: y = torch.cumsum(x * mask, dim=1), shape (B, D), fp32.
//
// Mask gating, scratch management, and the scan are implemented by one
// verified Kuiper entry point. This bridge only validates the ABI, selects the
// input device, makes one Kuiper call, and wraps the returned allocation.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_MaskedCumSum.h"
#include "Kuiper_KB_MaskedCumSum.cu"

static_assert(sizeof(bool) == sizeof(uint8_t));

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t) 2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t) 1024;
static constexpr int64_t KUIPER_MAX_MAP_CELLS =
    KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS;

torch::Tensor kuiper_masked_cumsum_dim1_cuda(torch::Tensor X,
                                             torch::Tensor Mask)
{
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 &&
                    X.dim() == 2,
                "kuiper_masked_cumsum_dim1: expected 2-D float32 CUDA tensor");
    TORCH_CHECK(
        Mask.is_cuda() && Mask.scalar_type() == torch::kBool &&
            Mask.dim() == 2 && Mask.size(0) == X.size(0) &&
            Mask.size(1) == X.size(1),
        "kuiper_masked_cumsum_dim1: expected same-shape CUDA bool mask");
    TORCH_CHECK(X.device() == Mask.device(),
                "kuiper_masked_cumsum_dim1: inputs must be on the same device");
    TORCH_CHECK(X.is_contiguous() && Mask.is_contiguous(),
                "kuiper_masked_cumsum_dim1: inputs must be contiguous");
    int64_t B = X.size(0), D = X.size(1);
    TORCH_CHECK(B > 0 && D > 0 && B <= (int64_t) UINT32_MAX &&
                    D <= (int64_t) UINT32_MAX &&
                    B <= KUIPER_MAX_MAP_CELLS / D && B <= KUIPER_MAX_BLOCKS,
                "kuiper_masked_cumsum_dim1: shape out of range");
    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_MaskedCumSum_masked_cumsum_fw_f32(
        (uint32_t) B, (uint32_t) D, X.data_ptr<float>(),
        reinterpret_cast<uint8_t *>(Mask.data_ptr<bool>()));
    return torch::from_blob(
        out, {B, D}, [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_masked_cumsum_dim1", &kuiper_masked_cumsum_dim1_cuda,
          "Kuiper verified masked cumulative sum along dim=1 of a 2-D tensor");
}
