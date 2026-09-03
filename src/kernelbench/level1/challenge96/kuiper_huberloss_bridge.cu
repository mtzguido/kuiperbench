// Bridge for KernelBench L1 #96: Smooth L1 (Huber) Loss.
//
// PyTorch reference: F.smooth_l1_loss(predictions, targets)
//                    (defaults: beta=1.0, reduction='mean')
//   loss_i = 0.5 * (p_i - t_i)^2     if |p_i - t_i| < 1
//          = |p_i - t_i| - 0.5       otherwise
//   output = mean_i loss_i
//
// The public Kuiper entry preserves both inputs, owns scratch/reduction/mean,
// and returns an allocated GPU scalar.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_HuberLoss.h"
#include "Kuiper_KB_HuberLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t) 2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t) 1024;

torch::Tensor kuiper_huberloss_cuda(torch::Tensor predictions,
                                    torch::Tensor targets)
{
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda() &&
                    predictions.scalar_type() == torch::kFloat32 &&
                    targets.scalar_type() == torch::kFloat32 &&
                    predictions.sizes() == targets.sizes(),
                "kuiper_huberloss: expected equal-shaped float32 CUDA tensors");
    TORCH_CHECK(predictions.device() == targets.device(),
                "kuiper_huberloss: inputs must be on the same CUDA device");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(),
                "kuiper_huberloss: inputs must be contiguous");
    int64_t N = predictions.numel();
    TORCH_CHECK(N > 0 && N <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS &&
                    N + KUIPER_MAX_THREADS <= (int64_t) UINT32_MAX,
                "kuiper_huberloss: shape out of range");
    const c10::cuda::CUDAGuard device_guard(predictions.device());
    float *out = Kuiper_KB_HuberLoss_huber_loss_fw_f32(
        (uint32_t) N, predictions.data_ptr<float>(), targets.data_ptr<float>());
    return torch::from_blob(
        out, c10::IntArrayRef{}, [](void *p) { cudaFree(p); },
        predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_huberloss", &kuiper_huberloss_cuda,
          "Kuiper verified Smooth L1 (Huber) loss");
}
