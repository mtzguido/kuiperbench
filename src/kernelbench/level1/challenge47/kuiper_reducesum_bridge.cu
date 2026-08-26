// Bridge for KernelBench L1 #47: sum reduction over the middle dim of
// a (B, D, M) tensor.
//
// PyTorch: y = torch.sum(x, dim=1, keepdim=True), shape (B, 1, M).
//
// Kuiper: factor (B, D, M) row-major as Array2 with layout
//     l2_bcm_pages B M D
// whose imap (r, ci) -> (r/M)*D*M + ci*M + r%M  exactly matches the
// physical row-major (B, D, M) layout.  Row r = b*M + j carries the
// length-D slice x[b,:,j].  One launch of [reduce_batched_block]
// produces y[b*M+j] ≈ Σ_k x[b,k,j].
#include <torch/extension.h>
#include "Kuiper_KB_ReduceSum.h"
#include "Kuiper_KB_ReduceSum.cu"

// max_blocks from Kuiper.Base = 2^21
static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
// max_threads from Kuiper.Base = 1024
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_reducesum_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 3,
                "kuiper_reducesum_dim1: expected 3-D float32 CUDA tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), D = Xc.size(1), M = Xc.size(2);
    TORCH_CHECK(B > 0 && D > 0 && M > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && M <= (int64_t)UINT32_MAX
                && B * M <= (int64_t)UINT32_MAX
                && M * D <= (int64_t)UINT32_MAX
                && B * M * D <= (int64_t)UINT32_MAX
                && B * M <= KUIPER_MAX_BLOCKS
                && D + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX,
                "kuiper_reducesum_dim1: shape out of range");
    auto Y = torch::empty({B, 1, M}, Xc.options());
    Kuiper_KB_ReduceSum_reduce_sum_fw_f32(
        (uint32_t)B, (uint32_t)M, (uint32_t)D,
        Xc.data_ptr<float>(), Y.data_ptr<float>());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_reducesum_dim1", &kuiper_reducesum_dim1_cuda,
          "Kuiper verified sum reduction over dim=1 of a 3-D tensor");
}
