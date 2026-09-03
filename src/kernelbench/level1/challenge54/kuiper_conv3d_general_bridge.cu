#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_Conv3DAlloc.h"
#include "Kuiper_KB_Conv3DAlloc.cu"

static torch::Tensor kuiper_conv3d_general_cuda(
    torch::Tensor X, torch::Tensor W, c10::optional<torch::Tensor> Bias_opt,
    int64_t stride_d, int64_t stride_h, int64_t stride_w, int64_t pad_d,
    int64_t pad_h, int64_t pad_w, int64_t dil_d, int64_t dil_h, int64_t dil_w,
    int64_t groups)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.device() == W.device(),
                "kuiper_conv3d: X and W must share a CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32,
                "kuiper_conv3d: X and W must be float32");
    TORCH_CHECK(X.dim() == 5 && W.dim() == 5 && X.is_contiguous() &&
                    W.is_contiguous(),
                "kuiper_conv3d: expected contiguous NCDHW/OIDHW tensors");
    TORCH_CHECK(stride_d == stride_h && stride_h == stride_w && stride_d >= 1 &&
                    pad_d == pad_h && pad_h == pad_w && pad_d >= 0 &&
                    dil_d == 1 && dil_h == 1 && dil_w == 1 && groups == 1 &&
                    stride_d <= (int64_t) UINT32_MAX &&
                    pad_d <= (int64_t) UINT32_MAX,
                "kuiper_conv3d: unsupported parameters or uint32 ABI overflow");

    const int64_t B = X.size(0), Cin = X.size(1), Din = X.size(2);
    const int64_t Hin = X.size(3), Win = X.size(4), Cout = W.size(0);
    const int64_t WCin = W.size(1), Kd = W.size(2);
    const int64_t Kh = W.size(3), Kw = W.size(4);
    TORCH_CHECK(
        B > 0 && Cin > 0 && Din > 0 && Hin > 0 && Win > 0 && Cout > 0 &&
            Kd > 0 && Kh > 0 && Kw > 0 && B <= (int64_t) UINT32_MAX &&
            Cin <= (int64_t) UINT32_MAX && Din <= (int64_t) UINT32_MAX &&
            Hin <= (int64_t) UINT32_MAX && Win <= (int64_t) UINT32_MAX &&
            Cout <= (int64_t) UINT32_MAX && Kd <= (int64_t) UINT32_MAX &&
            Kh <= (int64_t) UINT32_MAX && Kw <= (int64_t) UINT32_MAX &&
            Cin == WCin,
        "kuiper_conv3d: invalid shapes or uint32 ABI overflow");

    const bool has_bias = Bias_opt.has_value() && Bias_opt->defined();
    if (has_bias) {
        const auto &Bias = *Bias_opt;
        TORCH_CHECK(Bias.is_cuda() && Bias.device() == X.device() &&
                        Bias.scalar_type() == torch::kFloat32 &&
                        Bias.dim() == 1 && Bias.size(0) == Cout &&
                        Bias.is_contiguous(),
                    "kuiper_conv3d: invalid bias");
    }

    const c10::cuda::CUDAGuard device_guard(X.device());
    Kuiper_KB_Conv3DAlloc_conv3d_raw_result r;
    if (has_bias) {
        r = Kuiper_KB_Conv3DAlloc_conv3d_raw_alloc_bias_f32(
            (uint32_t) B, (uint32_t) Cin, (uint32_t) Din, (uint32_t) Hin,
            (uint32_t) Win, (uint32_t) Cout, (uint32_t) Kd, (uint32_t) Kh,
            (uint32_t) Kw, (uint32_t) stride_d, (uint32_t) pad_d,
            X.data_ptr<float>(), W.data_ptr<float>(),
            Bias_opt->data_ptr<float>());
    } else {
        r = Kuiper_KB_Conv3DAlloc_conv3d_raw_alloc_zero_f32(
            (uint32_t) B, (uint32_t) Cin, (uint32_t) Din, (uint32_t) Hin,
            (uint32_t) Win, (uint32_t) Cout, (uint32_t) Kd, (uint32_t) Kh,
            (uint32_t) Kw, (uint32_t) stride_d, (uint32_t) pad_d,
            X.data_ptr<float>(), W.data_ptr<float>());
    }

    return torch::from_blob(
        r.output,
        {B, Cout, (int64_t) r.d_out, (int64_t) r.h_out,
         (int64_t) r.w_out},
        [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_conv3d_general", &kuiper_conv3d_general_cuda,
          "Kuiper verified Conv3D forward");
}
