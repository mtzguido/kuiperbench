// Bridge for KernelBench L1 #39: L2 normalization along dim=1.  In-place.
#include <torch/extension.h>
#include "Kuiper_KB_L2Norm.h"
#include "Kuiper_KB_L2Norm.cu"

torch::Tensor kuiper_l2norm_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.dim() == 2 && X.scalar_type() == torch::kFloat32,
                "kuiper_l2norm: expected 2D CUDA float32 tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), D = Xc.size(1);
    TORCH_CHECK(B > 0 && D > 0 && B <= (int64_t)UINT32_MAX
                && D + 1024 <= (int64_t)UINT32_MAX,
                "kuiper_l2norm: shape out of range");
    Kuiper_KB_L2Norm_l2norm_fw_f32((uint32_t)B, (uint32_t)D,
                                   Xc.data_ptr<float>());
    return Xc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_l2norm", &kuiper_l2norm_cuda,
          "Kuiper verified L2 normalization along dim=1");
}
