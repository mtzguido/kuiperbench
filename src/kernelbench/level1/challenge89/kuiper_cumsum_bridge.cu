// Bridge for KernelBench L1 #89: cumulative sum along dim=1 of a (B, D)
// row-major float32 tensor.
//
// PyTorch: y = torch.cumsum(x, dim=1), shape (B, D), float32.
//
// Kuiper: Array2 of shape (B, D) under l2_row_major.  One launch of
// the row-per-block sequential inclusive prefix-scan primitive
// [Kuiper.Kernel.Scan1D.RowBlock.scan1d_inclusive_rowblock] performs
// the full cumulative sum, with rows scanned in parallel across
// blocks (1 thread / block, sequential within a row).  Total work
// O(B * D), parallelism = B blocks.
#include <torch/extension.h>
#include "Kuiper_KB_CumSum.h"
#include "Kuiper_KB_CumSum.cu"

// max_blocks from Kuiper.Base = 2^21
static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t)2097152;

torch::Tensor kuiper_cumsum_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 2,
                "kuiper_cumsum_dim1: expected 2-D float32 CUDA tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), D = Xc.size(1);
    TORCH_CHECK(B > 0 && D > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && B * D <= (int64_t)UINT32_MAX
                && B <= KUIPER_MAX_BLOCKS,
                "kuiper_cumsum_dim1: shape out of range");
    auto Y = torch::empty_like(Xc);
    Kuiper_KB_CumSum_cumsum_fw_f32(
        (uint32_t)B, (uint32_t)D,
        Xc.data_ptr<float>(), Y.data_ptr<float>());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_cumsum_dim1", &kuiper_cumsum_dim1_cuda,
          "Kuiper verified cumulative sum along dim=1 of a 2-D tensor");
}
