// Bridge for KernelBench L1 #98: KLDivLoss (reduction='batchmean').
//
// PyTorch reference:
//   F.kl_div(torch.log(predictions), targets, reduction='batchmean')
//   = sum(targets * (log(targets) - log(predictions))) / batch_size
//
// The verified kernel returns the batch mean directly.
#include <torch/extension.h>
#include "Kuiper_KB_KLDivLoss.h"
#include "Kuiper_KB_KLDivLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_kldivloss_cuda(torch::Tensor predictions,
                                    torch::Tensor targets,
                                    int64_t batch_size) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda()
                && predictions.scalar_type() == torch::kFloat32
                && targets.scalar_type() == torch::kFloat32
                && predictions.dim() == 1 && targets.dim() == 1
                && predictions.size(0) == targets.size(0),
                "kuiper_kldivloss: expected two equal-length 1-D float32 CUDA tensors");
    TORCH_CHECK(batch_size > 0 && batch_size <= (int64_t)UINT32_MAX,
                "kuiper_kldivloss: batch_size outside the verified size domain");
    auto P = predictions.contiguous();
    auto T = targets.contiguous();
    int64_t N = P.size(0);
    TORCH_CHECK(N > 0
                && N <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && N <= (int64_t)UINT32_MAX,
                "kuiper_kldivloss: shape out of range");
    TORCH_CHECK((P > 0).all().item<bool>() && (T > 0).all().item<bool>(),
                "kuiper_kldivloss: predictions and targets must be positive");
    float mean = Kuiper_KB_KLDivLoss_kl_div_fw_f32(
        (uint32_t)N, (uint32_t)batch_size,
        P.data_ptr<float>(), T.data_ptr<float>());
    auto out = torch::tensor(mean, torch::dtype(torch::kFloat32).device(torch::kCPU));
    return out.to(predictions.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_kldivloss", &kuiper_kldivloss_cuda,
          "Kuiper verified KL divergence (batchmean)");
}
