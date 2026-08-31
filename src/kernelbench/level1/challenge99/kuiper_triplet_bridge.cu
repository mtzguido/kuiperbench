// Bridge for KernelBench L1 #99: Triplet Margin Loss.
//
// PyTorch reference (margin=1.0, p=2, eps=1e-6, reduction='mean'):
//   dist_eps(x,y) = ||x - y + eps * 1||_2
//   L = mean_b max(0, dist_eps(a_b,p_b) - dist_eps(a_b,n_b) + margin)
//
// The verified kernel runs the full per-row distance + reduction +
// margin step + B-reduce + scalar multiply by the internally computed [1/B],
// returning the mean scalar directly (no length-1 output buffer).
#include <torch/extension.h>
#include "Kuiper_KB_TripletMarginLoss.h"
#include "Kuiper_KB_TripletMarginLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_triplet_cuda(torch::Tensor anchor,
                                  torch::Tensor positive,
                                  torch::Tensor negative,
                                  double margin) {
    TORCH_CHECK(anchor.is_cuda() && positive.is_cuda() && negative.is_cuda()
                && anchor.scalar_type()   == torch::kFloat32
                && positive.scalar_type() == torch::kFloat32
                && negative.scalar_type() == torch::kFloat32
                && anchor.dim()   == 2
                && positive.dim() == 2
                && negative.dim() == 2
                && anchor.size(0) == positive.size(0)
                && anchor.size(0) == negative.size(0)
                && anchor.size(1) == positive.size(1)
                && anchor.size(1) == negative.size(1),
                "kuiper_triplet: expected three (B, D) float32 CUDA tensors");
    auto A = anchor.contiguous();
    auto P = positive.contiguous();
    auto N = negative.contiguous();
    int64_t B = A.size(0), D = A.size(1);
    int64_t BD = B * D;
    TORCH_CHECK(B > 0 && D > 0
                && B  <= (int64_t)UINT32_MAX
                && D  <= (int64_t)UINT32_MAX
                && BD <= (int64_t)UINT32_MAX
                && B  <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && D  <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && BD <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && B + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX
                && D + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX,
                "kuiper_triplet: shape out of range");
    constexpr float eps = 1.0e-6f;
    auto res = Kuiper_KB_TripletMarginLoss_triplet_fw_f32(
        (uint32_t)B, (uint32_t)D,
        (float)margin, eps,
        A.data_ptr<float>(),
        P.data_ptr<float>(),
        N.data_ptr<float>());
    auto out = torch::tensor(res, torch::dtype(torch::kFloat32).device(torch::kCPU));
    return out.to(anchor.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_triplet", &kuiper_triplet_cuda,
          "Kuiper verified Triplet Margin Loss (p=2, eps=1e-6, reduction=mean)");
}
