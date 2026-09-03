// Bridge for KernelBench L1 #100: Hinge Loss.
//
// PyTorch reference: torch.mean(torch.clamp(1 - predictions * targets, min=0))
//
// The single public Kuiper entry accepts predictions (B,N) and targets (N),
// proves the broadcast target[j] across every row, preserves both inputs, and
// returns a freshly allocated one-element GPU buffer.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "Kuiper_KB_HingeLoss.h"
#include "Kuiper_KB_HingeLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_hingeloss_cuda(torch::Tensor predictions,
                                    torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda()
                && predictions.scalar_type() == torch::kFloat32
                && targets.scalar_type() == torch::kFloat32
                && predictions.dim() == 2 && targets.dim() == 1,
                "kuiper_hingeloss: expected predictions (B,N) and targets (N) "
                "as float32 CUDA tensors");
    TORCH_CHECK(targets.size(0) == predictions.size(1),
                "kuiper_hingeloss: targets length must equal predictions.size(1)");
    TORCH_CHECK(predictions.device() == targets.device(),
                "kuiper_hingeloss: inputs must be on the same CUDA device");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(),
                "kuiper_hingeloss: inputs must be contiguous");

    const int64_t B = predictions.size(0);
    const int64_t N = predictions.size(1);
    using wide = unsigned __int128;
    const wide elems = (wide)B * (wide)N;
    const wide launch_limit =
        (wide)KUIPER_MAX_BLOCKS * (wide)KUIPER_MAX_THREADS;
    TORCH_CHECK(B > 0 && N > 0
                && (wide)B <= (wide)UINT32_MAX
                && (wide)N <= (wide)UINT32_MAX
                && elems <= launch_limit
                && elems + (wide)KUIPER_MAX_THREADS <= (wide)UINT32_MAX,
                "kuiper_hingeloss: shape out of range");

    const c10::cuda::CUDAGuard device_guard(predictions.device());
    float *out = Kuiper_KB_HingeLoss_hinge_loss_broadcast_f32(
        (uint32_t)B, (uint32_t)N,
        predictions.data_ptr<float>(), targets.data_ptr<float>());
    return torch::from_blob(
        out, c10::IntArrayRef{}, [](void *p) { cudaFree(p); },
        predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_hingeloss", &kuiper_hingeloss_cuda,
          "Kuiper verified broadcast Hinge Loss");
}
