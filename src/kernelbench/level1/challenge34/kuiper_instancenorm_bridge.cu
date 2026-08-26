// Bridge for KernelBench L1 #34: InstanceNorm2d (no affine).  In-place.
//
// View x : (B, C, H, W) row-major as a flat (B*C, H*W) tensor and run
// row-wise mean/variance normalisation: per (n, c) slice over (H, W),
// compute (x - mean) / sqrt(var + eps).  eps default 1e-5.
#include <torch/extension.h>
#include "Kuiper_KB_MeanVarNorm.h"
#include "Kuiper_KB_MeanVarNorm.cu"

torch::Tensor kuiper_instancenorm_cuda(torch::Tensor X, double eps) {
    TORCH_CHECK(X.is_cuda() && X.dim() == 4 && X.scalar_type() == torch::kFloat32,
                "kuiper_instancenorm: expected 4D CUDA float32 tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), C = Xc.size(1), H = Xc.size(2), W = Xc.size(3);
    int64_t rows = B * C, d = H * W;
    TORCH_CHECK(rows > 0 && d > 0 && rows <= (int64_t)UINT32_MAX
                && d + 1024 <= (int64_t)UINT32_MAX
                && rows * d <= (int64_t)UINT32_MAX,
                "kuiper_instancenorm: shape out of range");
    Kuiper_KB_MeanVarNorm_mean_var_norm_fw_f32(
        (uint32_t)rows, (uint32_t)d, (float)eps,
        Xc.data_ptr<float>());
    return Xc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_instancenorm", &kuiper_instancenorm_cuda,
          "Kuiper verified InstanceNorm2d (no affine), in place");
}
