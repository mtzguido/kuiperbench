// Bridge for KernelBench L1 #100: Hinge Loss.
//
// PyTorch reference: torch.mean(torch.clamp(1 - predictions * targets, min=0))
//
// Verified pipeline (see Kuiper.Spec.HingeLoss.real_hinge):
//   1. predictions[i] := max(0, 1 - predictions[i] * targets[i])
//   2. tree_reduce_sum -> host scalar [s]
//   3. on-device division [res := s / n]; returned as a scalar.
#include <torch/extension.h>
#include "Kuiper_KB_HingeLoss.h"
#include "Kuiper_KB_HingeLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_hingeloss_cuda(torch::Tensor predictions,
                                    torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda()
                && predictions.scalar_type() == torch::kFloat32
                && targets.scalar_type() == torch::kFloat32
                && predictions.dim() == 1 && targets.dim() == 1
                && predictions.size(0) == targets.size(0),
                "kuiper_hingeloss: expected two equal-length 1-D float32 CUDA tensors");
    auto P = predictions.contiguous();
    auto T = targets.contiguous();
    int64_t N = P.size(0);
    TORCH_CHECK(N > 0
                && N <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && N <= (int64_t)UINT32_MAX,
                "kuiper_hingeloss: shape out of range");
    auto res = Kuiper_KB_HingeLoss_hinge_loss_fw_f32(
        (uint32_t)N,
        P.data_ptr<float>(), T.data_ptr<float>());
    auto out = torch::tensor(res, torch::dtype(torch::kFloat32).device(torch::kCPU));
    return out.to(predictions.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_hingeloss", &kuiper_hingeloss_cuda,
          "Kuiper verified Hinge Loss (mean of max(0, 1 - p*t))");
}
