// Bridge for KernelBench L1 #91: reverse cumulative sum along dim=1.
//
// PyTorch: y = torch.cumsum(x.flip(1), dim=1).flip(1), shape (B, D), fp32.
//
// Strategy: the *flips* are pure index permutations (no arithmetic); the
// scan is the part that needs verification.  We forward the flipped
// input to the verified Kuiper cumsum kernel and flip the result back
// host-side.  The C++ side only ever calls the verified scan.
#include <torch/extension.h>
#include "Kuiper_KB_CumSum.h"
#include "Kuiper_KB_CumSum.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t)2097152;

torch::Tensor kuiper_cumsum_reverse_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 2,
                "kuiper_cumsum_reverse_dim1: expected 2-D float32 CUDA tensor");
    auto Xf = X.flip(1).contiguous();
    int64_t B = Xf.size(0), D = Xf.size(1);
    TORCH_CHECK(B > 0 && D > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && B * D <= (int64_t)UINT32_MAX
                && B <= KUIPER_MAX_BLOCKS,
                "kuiper_cumsum_reverse_dim1: shape out of range");
    auto Y = torch::empty_like(Xf);
    Kuiper_KB_CumSum_cumsum_fw_f32(
        (uint32_t)B, (uint32_t)D,
        Xf.data_ptr<float>(), Y.data_ptr<float>());
    return Y.flip(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_cumsum_reverse_dim1", &kuiper_cumsum_reverse_dim1_cuda,
          "Kuiper verified reverse cumulative sum along dim=1 of a 2-D tensor");
}
