// Bridge for KernelBench L1 #36: RMSNorm along dim=1.
//
// Input X has shape (B, C, H, W).  RMSNorm divides by
//     rms[b,h,w] = sqrt( mean_c X[b,c,h,w]^2 + eps )
//
// Kuiper view: the contiguous (B, C, H, W) row-major buffer is treated
// as Array2 with layout `l2_bcm_pages B HW C` whose imap
//     (r, ci) -> (r/HW)*C*HW + ci*HW + r%HW
// matches the physical row-major (B, C, H, W) layout exactly.
// Exactly 3 GPU kernel launches: reduce_batched, map_gpu, row_scale.
#include <torch/extension.h>
#include <cmath>
#include "Kuiper_KB_RMSNorm.h"
#include "Kuiper_KB_RMSNorm.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_rmsnorm_cuda(torch::Tensor X, double eps) {
    TORCH_CHECK(X.is_cuda() && X.dim() >= 2 && X.scalar_type() == torch::kFloat32,
                "kuiper_rmsnorm: expected >=2D CUDA float32 tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), C = Xc.size(1);
    int64_t HW = 1;
    for (int i = 2; i < Xc.dim(); ++i) HW *= Xc.size(i);
    TORCH_CHECK(B > 0 && HW > 0 && C > 0
                && B  <= (int64_t)UINT32_MAX
                && HW <= (int64_t)UINT32_MAX
                && C  <= (int64_t)UINT32_MAX
                && B * HW <= (int64_t)UINT32_MAX
                && HW * C <= (int64_t)UINT32_MAX
                && B * HW * C <= (int64_t)UINT32_MAX
                && B * HW * C <= KUIPER_MAX_NTHR,
                "kuiper_rmsnorm: shape out of range");
    float eps_f = (float)eps;
    TORCH_CHECK(std::isfinite(eps_f) && eps_f > 0.0f,
                "kuiper_rmsnorm: eps must remain finite and positive in float32");
    Kuiper_KB_RMSNorm_rmsnorm_fw_f32(
        (uint32_t)B, (uint32_t)HW, (uint32_t)C,
        eps_f,
        Xc.data_ptr<float>());
    return Xc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_rmsnorm", &kuiper_rmsnorm_cuda,
          "Kuiper verified RMSNorm along dim=1 (3 GPU launches)");
}
