// Bridge for KernelBench L1 #93: masked cumulative sum along dim=1.
//
// PyTorch: y = torch.cumsum(x * mask, dim=1), shape (B, D), fp32.
//
// We compute x_masked = x * mask.float() with a PyTorch elementwise op
// (no race-prone arithmetic, just a pointwise gate) and then feed the
// result into the verified Kuiper cumsum kernel.  The scan — the only
// part with non-trivial loop / accumulator semantics — is done by the
// verified kernel.
#include <torch/extension.h>
#include "Kuiper_KB_CumSum.h"
#include "Kuiper_KB_CumSum.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t)2097152;

torch::Tensor kuiper_masked_cumsum_dim1_cuda(torch::Tensor X, torch::Tensor Mask) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 2,
                "kuiper_masked_cumsum_dim1: expected 2-D float32 CUDA tensor");
    TORCH_CHECK(Mask.is_cuda() && Mask.dim() == 2
                && Mask.size(0) == X.size(0) && Mask.size(1) == X.size(1),
                "kuiper_masked_cumsum_dim1: mask shape mismatch");
    auto Mf = Mask.to(torch::kFloat32);
    auto Xm = (X * Mf).contiguous();
    int64_t B = Xm.size(0), D = Xm.size(1);
    TORCH_CHECK(B > 0 && D > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && B * D <= (int64_t)UINT32_MAX
                && B <= KUIPER_MAX_BLOCKS,
                "kuiper_masked_cumsum_dim1: shape out of range");
    auto Y = torch::empty_like(Xm);
    Kuiper_KB_CumSum_cumsum_fw_f32(
        (uint32_t)B, (uint32_t)D,
        Xm.data_ptr<float>(), Y.data_ptr<float>());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_masked_cumsum_dim1", &kuiper_masked_cumsum_dim1_cuda,
          "Kuiper verified masked cumulative sum along dim=1 of a 2-D tensor");
}
