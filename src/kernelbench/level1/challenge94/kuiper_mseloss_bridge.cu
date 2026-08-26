// Bridge for KernelBench L1 #94: Mean Squared Error.
//
// PyTorch reference: torch.mean((predictions - targets) ** 2).
//
// Verified pipeline (see Kuiper.Spec.MSELoss):
//   1. predictions[i] := (predictions[i] - targets[i])^2  (in-place)
//   2. tree_reduce_sum -> host scalar [s]
#include <torch/extension.h>
#include "Kuiper_KB_MSELoss.h"
#include "Kuiper_KB_MSELoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_mseloss_cuda(torch::Tensor predictions,
                                  torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda()
                && predictions.scalar_type() == torch::kFloat32
                && targets.scalar_type() == torch::kFloat32
                && predictions.dim() == 1 && targets.dim() == 1
                && predictions.size(0) == targets.size(0),
                "kuiper_mseloss: expected two equal-length 1-D float32 CUDA tensors");
    auto P = predictions.contiguous();
    auto T = targets.contiguous();
    int64_t N = P.size(0);
    TORCH_CHECK(N > 0
                && N <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && N + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX,
                "kuiper_mseloss: shape out of range");
    auto res = Kuiper_KB_MSELoss_mse_loss_fw_f32(
        (uint32_t)N,
        P.data_ptr<float>(), T.data_ptr<float>());
    auto out = torch::tensor(res, torch::dtype(torch::kFloat32).device(torch::kCPU));
    return out.to(predictions.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_mseloss", &kuiper_mseloss_cuda,
          "Kuiper verified Mean Squared Error loss");
}
