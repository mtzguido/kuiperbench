// Bridge for KernelBench L1 #92: exclusive cumulative sum along dim=1 of
// a (B, D) row-major float32 tensor.
//
// PyTorch reference:
//   cumsum = torch.cumsum(x[:, :-1], dim=1)
//   y = cat(zeros[:, 0:1], cumsum, dim=1)            # shape (B, D)
// Equivalently: y[b, j] = sum_{i < j} x[b, i], with y[b, 0] = 0.
//
// Kuiper: Array2 of shape (B, D) under l2_row_major.  One launch of
// the row-per-block sequential *exclusive* prefix-scan primitive
// [Kuiper.Kernel.Scan1D.RowBlockExcl.scan1d_exclusive_rowblock]
// performs the full exclusive cumulative sum directly, with rows
// scanned in parallel across blocks (1 thread / block, sequential
// within a row).  The exclusive transform is performed entirely
// inside the verified kernel — each output cell receives the running
// accumulator BEFORE the cell's own input element is folded in — so
// there is NO host-side subtraction or shift.  Total work O(B * D),
// parallelism = B blocks.
//
// The kernel's verified postcondition (cumsum_exclusive_post) ties
// output[b, j] (up to fp32 %~ approximation) to the real-arithmetic
// exclusive prefix sum rsum(x[b, 0..j)).  This bridge therefore does
// only dimension-contract checks and plumbing.
#include <torch/extension.h>
#include "Kuiper_KB_CumSumExclusive.h"
#include "Kuiper_KB_CumSumExclusive.cu"

// max_blocks from Kuiper.Base = 2^21
static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t)2097152;

torch::Tensor kuiper_cumsum_exclusive_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 2,
                "kuiper_cumsum_exclusive_dim1: expected 2-D float32 CUDA tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), D = Xc.size(1);
    TORCH_CHECK(B > 0 && D > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && B * D <= (int64_t)UINT32_MAX
                && B <= KUIPER_MAX_BLOCKS,
                "kuiper_cumsum_exclusive_dim1: shape out of range");
    auto Y = torch::empty_like(Xc);
    Kuiper_KB_CumSumExclusive_cumsum_exclusive_fw_f32(
        (uint32_t)B, (uint32_t)D,
        Xc.data_ptr<float>(), Y.data_ptr<float>());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_cumsum_exclusive_dim1", &kuiper_cumsum_exclusive_dim1_cuda,
          "Kuiper verified exclusive cumulative sum along dim=1 of a 2-D tensor");
}
