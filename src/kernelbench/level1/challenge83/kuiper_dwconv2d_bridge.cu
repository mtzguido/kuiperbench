#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_DepthwiseConv2D.h"
#include "Kuiper_KB_DepthwiseConv2D.cu"

static torch::Tensor kuiper_dwconv2d_cuda(torch::Tensor X, torch::Tensor W,
                                          c10::optional<torch::Tensor> Bias_opt,
                                          int64_t sh, int64_t sw, int64_t ph,
                                          int64_t pw, int64_t dh, int64_t dw,
                                          int64_t groups)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.device() == W.device(),
                "kuiper_dwconv2d: X and W must share a CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32,
                "kuiper_dwconv2d: X and W must be float32");
    TORCH_CHECK(X.dim() == 4 && W.dim() == 4 && X.is_contiguous() &&
                    W.is_contiguous(),
                "kuiper_dwconv2d: expected contiguous NCHW/OIHW tensors");
    TORCH_CHECK(
        sh == sw && sh >= 1 && ph == pw && ph >= 0 && dh == 1 && dw == 1 &&
            sh <= (int64_t) UINT32_MAX && ph <= (int64_t) UINT32_MAX,
        "kuiper_dwconv2d: unsupported parameters or uint32 ABI overflow");

    const int64_t B = X.size(0), C = X.size(1);
    const int64_t Hin = X.size(2), Win = X.size(3);
    const int64_t Cw = W.size(0), Cmul = W.size(1);
    const int64_t Kh = W.size(2), Kw = W.size(3);
    TORCH_CHECK(B > 0 && C > 0 && Hin > 0 && Win > 0 && Kh > 0 && Kw > 0 &&
                    B <= (int64_t) UINT32_MAX && C <= (int64_t) UINT32_MAX &&
                    Hin <= (int64_t) UINT32_MAX &&
                    Win <= (int64_t) UINT32_MAX && Kh <= (int64_t) UINT32_MAX &&
                    Kw <= (int64_t) UINT32_MAX && C == Cw && Cmul == 1 &&
                    groups == C,
                "kuiper_dwconv2d: invalid shapes, groups, or uint32 overflow");

    const bool has_bias = Bias_opt.has_value() && Bias_opt->defined();
    if (has_bias) {
        const auto &Bias = *Bias_opt;
        TORCH_CHECK(Bias.is_cuda() && Bias.device() == X.device() &&
                        Bias.scalar_type() == torch::kFloat32 &&
                        Bias.dim() == 1 && Bias.size(0) == C &&
                        Bias.is_contiguous(),
                    "kuiper_dwconv2d: invalid bias");
    }

    const c10::cuda::CUDAGuard device_guard(X.device());
    Prims_dtuple2__uint32_t_Prims_dtuple2__uint32_t__float_ r;
    if (has_bias) {
        r = Kuiper_KB_DepthwiseConv2D_dwconv2d_raw_alloc_bias_f32(
            (uint32_t) B, (uint32_t) C, (uint32_t) Hin, (uint32_t) Win,
            (uint32_t) Kh, (uint32_t) Kw, (uint32_t) sh, (uint32_t) ph,
            X.data_ptr<float>(), W.data_ptr<float>(),
            Bias_opt->data_ptr<float>());
    } else {
        r = Kuiper_KB_DepthwiseConv2D_dwconv2d_raw_alloc_zero_f32(
            (uint32_t) B, (uint32_t) C, (uint32_t) Hin, (uint32_t) Win,
            (uint32_t) Kh, (uint32_t) Kw, (uint32_t) sh, (uint32_t) ph,
            X.data_ptr<float>(), W.data_ptr<float>());
    }

    return torch::from_blob(
        r.snd.snd, {B, C, (int64_t) r.fst, (int64_t) r.snd.fst},
        [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_dwconv2d", &kuiper_dwconv2d_cuda,
          "Kuiper verified depthwise Conv2D forward");
}
