// Bridge for KernelBench L1 #38: L1 normalization along dim=1.  In-place.
#include <torch/extension.h>
#include "Kuiper_KB_L1Norm.h"
#include "Kuiper_KB_L1Norm.cu"

torch::Tensor kuiper_l1norm_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.dim() == 2 && X.scalar_type() == torch::kFloat32,
                "kuiper_l1norm: expected 2D CUDA float32 tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), D = Xc.size(1);
    // Kernel requires B > 0, D > 0 and B*D <= max_blocks*max_threads = 2^31.
    TORCH_CHECK(B > 0 && D > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && B * D <= ((int64_t)1 << 31),
                "kuiper_l1norm: shape out of range");
    Kuiper_KB_L1Norm_l1norm_fw_f32((uint32_t)B, (uint32_t)D,
                                   Xc.data_ptr<float>());
    return Xc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_l1norm", &kuiper_l1norm_cuda,
          "Kuiper verified L1 normalization along dim=1");
}
