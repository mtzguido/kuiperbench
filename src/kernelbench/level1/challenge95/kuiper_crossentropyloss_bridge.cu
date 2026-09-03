// Bridge for KernelBench L1 #95: Cross-Entropy Loss (mean reduction).
//
// PyTorch reference:
//   torch.nn.functional.cross_entropy(predictions, targets)
//   = mean_b ( -log_softmax(predictions[b])[targets[b]] )
//
// The WHOLE computation is performed inside the verification boundary by
// the single verified entry [Kuiper.KB.CrossEntropyLoss.ce_loss_fw_f32]:
// per-row numerically-stable log-softmax + (negated) target gather,
// on-device reduce-sum over the B rows, and multiply by the verified
// reciprocal 1/B.  The returned scalar is VERIFIED as a function of ALL
// inputs (predictions + targets); its post is
// [Kuiper.Spec.CrossEntropyLoss.cross_entropy_post].
//
// The bridge only validates the verified entry's shape/layout/range
// preconditions, selects the CUDA device, and transfers ownership of the
// one-element GPU result returned by Kuiper.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_CrossEntropyLoss.h"
#include "Kuiper_KB_CrossEntropyLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_crossentropy_cuda(torch::Tensor predictions,
                                       torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda()
                && predictions.scalar_type() == torch::kFloat32
                && targets.scalar_type() == torch::kInt64
                && predictions.dim() == 2
                && targets.dim() == 1
                && predictions.size(0) == targets.size(0),
                "kuiper_crossentropy: expected (B, C) f32 predictions and (B,) "
                "int64 targets on CUDA");
    TORCH_CHECK(predictions.device() == targets.device(),
                "kuiper_crossentropy: inputs must be on the same CUDA device");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(),
                "kuiper_crossentropy: inputs must be contiguous");
    int64_t B = predictions.size(0);
    int64_t C = predictions.size(1);
    TORCH_CHECK(B > 0 && C > 0
                && C  <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && B  <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && C  <= (int64_t)UINT32_MAX
                && B  <= (int64_t)UINT32_MAX
                && B  <= (int64_t)UINT32_MAX / C
                && B + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX
                && C + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX,
                "kuiper_crossentropy: shape out of range");

    // This reduction is validation only; the loss arithmetic and the
    // int64-to-index conversion remain inside the verified entrypoint.
    TORCH_CHECK(targets.ge(0).logical_and(targets.lt(C)).all().item<bool>(),
                "kuiper_crossentropy: target out of range");

    const c10::cuda::CUDAGuard device_guard(predictions.device());
    float *out = Kuiper_KB_CrossEntropyLoss_ce_loss_fw_f32(
        (uint32_t)B, (uint32_t)C,
        predictions.data_ptr<float>(), targets.data_ptr<int64_t>());

    return torch::from_blob(
        out, c10::IntArrayRef{}, [](void *p) { cudaFree(p); },
        predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_crossentropy", &kuiper_crossentropy_cuda,
          "Kuiper verified CrossEntropyLoss (mean of -log_softmax(pred)[target])");
}
