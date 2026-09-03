// Bridge for KernelBench L1 #99: Triplet Margin Loss.
//
// PyTorch reference (margin=1.0, p=2, eps=1e-6, reduction='mean'):
//   dist_eps(x,y) = ||x - y + eps * 1||_2
//   L = mean_b max(0, dist_eps(a_b,p_b) - dist_eps(a_b,n_b) + margin)
//
// The verified kernel runs the full per-row distance + reduction +
// margin step + B-reduce + scalar multiply by the internally computed [1/B],
// then returns an owned one-element GPU buffer.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_TripletMarginLoss.h"
#include "Kuiper_KB_TripletMarginLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t) 2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t) 1024;

torch::Tensor kuiper_triplet_cuda(torch::Tensor anchor, torch::Tensor positive,
                                  torch::Tensor negative, double margin)
{
    TORCH_CHECK(anchor.is_cuda() && positive.is_cuda() && negative.is_cuda() &&
                    anchor.scalar_type() == torch::kFloat32 &&
                    positive.scalar_type() == torch::kFloat32 &&
                    negative.scalar_type() == torch::kFloat32 &&
                    anchor.dim() == 2 && positive.dim() == 2 &&
                    negative.dim() == 2 && anchor.size(0) == positive.size(0) &&
                    anchor.size(0) == negative.size(0) &&
                    anchor.size(1) == positive.size(1) &&
                    anchor.size(1) == negative.size(1),
                "kuiper_triplet: expected three (B, D) float32 CUDA tensors");
    TORCH_CHECK(anchor.device() == positive.device() &&
                    anchor.device() == negative.device(),
                "kuiper_triplet: inputs must be on the same CUDA device");
    TORCH_CHECK(anchor.is_contiguous() && positive.is_contiguous() &&
                    negative.is_contiguous(),
                "kuiper_triplet: inputs must be contiguous");
    int64_t B = anchor.size(0), D = anchor.size(1);
    TORCH_CHECK(B > 0 && D > 0 && B <= (int64_t) UINT32_MAX &&
                    D <= (int64_t) UINT32_MAX &&
                    B <= (int64_t) UINT32_MAX / D &&
                    B <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS &&
                    D <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS &&
                    B <= (KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS) / D &&
                    B + KUIPER_MAX_THREADS <= (int64_t) UINT32_MAX &&
                    D + KUIPER_MAX_THREADS <= (int64_t) UINT32_MAX,
                "kuiper_triplet: shape out of range");
    const c10::cuda::CUDAGuard device_guard(anchor.device());
    float *out = Kuiper_KB_TripletMarginLoss_triplet_fw_f32(
        (uint32_t) B, (uint32_t) D, margin,
        anchor.data_ptr<float>(), positive.data_ptr<float>(),
        negative.data_ptr<float>());
    return torch::from_blob(
        out, c10::IntArrayRef{}, [](void *p) { cudaFree(p); },
        anchor.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "kuiper_triplet", &kuiper_triplet_cuda,
        "Kuiper verified Triplet Margin Loss (p=2, eps=1e-6, reduction=mean)");
}
