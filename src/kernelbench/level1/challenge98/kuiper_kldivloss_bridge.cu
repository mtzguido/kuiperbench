// Bridge for KernelBench L1 #98: KLDivLoss (reduction='batchmean').
//
// PyTorch reference:
//   F.kl_div(torch.log(predictions), targets, reduction='batchmean')
//   = sum(targets * (log(targets) - log(predictions))) / batch_size
//
// The public Kuiper entry preserves both inputs and returns an allocated GPU
// scalar containing the verified batch mean.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_KLDivLoss.h"
#include "Kuiper_KB_KLDivLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t) 2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t) 1024;

torch::Tensor kuiper_kldivloss_cuda(torch::Tensor predictions,
                                    torch::Tensor targets, int64_t batch_size)
{
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda() &&
                    predictions.scalar_type() == torch::kFloat32 &&
                    targets.scalar_type() == torch::kFloat32 &&
                    predictions.sizes() == targets.sizes() &&
                    predictions.dim() >= 1,
                "kuiper_kldivloss: expected equal-shaped non-scalar float32 "
                "CUDA tensors");
    TORCH_CHECK(predictions.device() == targets.device(),
                "kuiper_kldivloss: inputs must be on the same CUDA device");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(),
                "kuiper_kldivloss: inputs must be contiguous");
    TORCH_CHECK(predictions.size(0) == batch_size,
                "kuiper_kldivloss: batch_size must equal predictions.size(0)");
    TORCH_CHECK(
        batch_size > 0 && batch_size <= (int64_t) UINT32_MAX,
        "kuiper_kldivloss: batch_size outside the verified size domain");
    int64_t N = predictions.numel();
    TORCH_CHECK(N > 0 && N <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS &&
                    N <= (int64_t) UINT32_MAX,
                "kuiper_kldivloss: shape out of range");
    const c10::cuda::CUDAGuard device_guard(predictions.device());
    TORCH_CHECK((predictions > 0).all().item<bool>() &&
                    (targets > 0).all().item<bool>(),
                "kuiper_kldivloss: predictions and targets must be positive");
    float *out = Kuiper_KB_KLDivLoss_kl_div_fw_f32(
        (uint32_t) N, (uint32_t) batch_size, predictions.data_ptr<float>(),
        targets.data_ptr<float>());
    return torch::from_blob(
        out, c10::IntArrayRef{}, [](void *p) { cudaFree(p); },
        predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_kldivloss", &kuiper_kldivloss_cuda,
          "Kuiper verified KL divergence (batchmean)");
}
