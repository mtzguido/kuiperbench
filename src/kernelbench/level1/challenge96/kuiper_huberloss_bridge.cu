// Bridge for KernelBench L1 #96: Smooth L1 (Huber) Loss.
//
// PyTorch reference: F.smooth_l1_loss(predictions, targets)
//                    (defaults: beta=1.0, reduction='mean')
//   loss_i = 0.5 * (p_i - t_i)^2     if |p_i - t_i| < 1
//          = |p_i - t_i| - 0.5       otherwise
//   output = mean_i loss_i
//
// Verified pipeline: see Kuiper.Spec.HuberLoss.huber_post.
#include <torch/extension.h>
#include "Kuiper_KB_HuberLoss.h"
#include "Kuiper_KB_HuberLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_huberloss_cuda(torch::Tensor predictions,
                                    torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda()
                && predictions.scalar_type() == torch::kFloat32
                && targets.scalar_type() == torch::kFloat32
                && predictions.dim() == 1 && targets.dim() == 1
                && predictions.size(0) == targets.size(0),
                "kuiper_huberloss: expected two equal-length 1-D float32 CUDA tensors");
    auto P = predictions.contiguous();
    auto T = targets.contiguous();
    int64_t N = P.size(0);
    TORCH_CHECK(N > 0
                && N <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && N + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX,
                "kuiper_huberloss: shape out of range");
    /* The kernel returns a scalar float. Wrap it in a tensor. */
    float res = Kuiper_KB_HuberLoss_huber_loss_fw_f32(
        (uint32_t)N,
        P.data_ptr<float>(), T.data_ptr<float>());
    auto out = torch::tensor(res, torch::dtype(torch::kFloat32).device(torch::kCPU));
    return out.to(predictions.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_huberloss", &kuiper_huberloss_cuda,
          "Kuiper verified Smooth L1 (Huber) loss");
}
