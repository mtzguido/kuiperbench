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
// This bridge therefore contains NO host reduction loop.  The only host
// loop is a pure bounds-check that establishes the kernel's precondition
// (every target index is in [0, C)); it performs no loss arithmetic.
#include <torch/extension.h>
#include "Kuiper_KB_CrossEntropyLoss.h"
#include "Kuiper_KB_CrossEntropyLoss.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_crossentropy_cuda(torch::Tensor predictions,
                                       torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda()
                && predictions.scalar_type() == torch::kFloat32
                && (targets.scalar_type() == torch::kInt64
                    || targets.scalar_type() == torch::kInt32)
                && predictions.dim() == 2
                && targets.dim() == 1
                && predictions.size(0) == targets.size(0),
                "kuiper_crossentropy: expected (B, C) f32 predictions and (B,) "
                "int targets on CUDA");
    auto P = predictions.contiguous();    // not modified (kernel copies rows)
    auto T = targets.contiguous().to(torch::kInt64);
    int64_t B = P.size(0);
    int64_t C = P.size(1);
    int64_t BC = B * C;
    TORCH_CHECK(B > 0 && C > 0
                && C  <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && B  <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && C  <= (int64_t)UINT32_MAX
                && B  <= (int64_t)UINT32_MAX
                && BC <= (int64_t)UINT32_MAX
                && B + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX
                && C + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX,
                "kuiper_crossentropy: shape out of range");

    // Bounds-check the targets (establishes the kernel's precondition that
    // every target index is in [0, C)).  This is a validation pass only --
    // it computes no part of the loss.
    auto T_cpu = T.to(torch::kCPU);
    const int64_t *Tcpu = T_cpu.data_ptr<int64_t>();
    for (int64_t b = 0; b < B; ++b) {
        int64_t tb = Tcpu[b];
        TORCH_CHECK(tb >= 0 && tb < C, "kuiper_crossentropy: target out of range");
    }

    // Materialize the targets as a contiguous (B,) uint32 device buffer
    // (the verified entry takes a [size_t]-modeled-as-uint32 index array).
    auto T32 = T.to(torch::kInt32).contiguous();
    uint32_t *Tp = reinterpret_cast<uint32_t *>(T32.data_ptr<int32_t>());

    float inv_b = Kuiper_KB_CrossEntropyLoss_ce_recip_f32((uint32_t)B);
    float mean = Kuiper_KB_CrossEntropyLoss_ce_loss_fw_f32(
        (uint32_t)B, (uint32_t)C, inv_b,
        P.data_ptr<float>(), Tp);

    auto out = torch::tensor(mean, torch::dtype(torch::kFloat32).device(torch::kCPU));
    return out.to(predictions.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_crossentropy", &kuiper_crossentropy_cuda,
          "Kuiper verified CrossEntropyLoss (mean of -log_softmax(pred)[target])");
}
