#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_ConvT3DGeneral.h"
#include "Kuiper_KB_ConvT3DGeneral.cu"

static torch::Tensor
kuiper_convt3d_general_cuda(torch::Tensor X, torch::Tensor W,
                            c10::optional<torch::Tensor> Bias_opt, int64_t sd,
                            int64_t sh, int64_t sw, int64_t pd, int64_t ph,
                            int64_t pw, int64_t opd, int64_t oph, int64_t opw,
                            int64_t dd, int64_t dh, int64_t dw, int64_t groups)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.device() == W.device(),
                "kuiper_convt3d: X and W must share a CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32,
                "kuiper_convt3d: X and W must be float32");
    TORCH_CHECK(X.dim() == 5 && W.dim() == 5 && X.is_contiguous() &&
                    W.is_contiguous(),
                "kuiper_convt3d: expected contiguous NCDHW/IODHW tensors");
    TORCH_CHECK(
        groups == 1 && sd >= 1 && sh >= 1 && sw >= 1 && pd >= 0 && ph >= 0 &&
            pw >= 0 && opd >= 0 && oph >= 0 && opw >= 0 && dd >= 1 && dh >= 1 &&
            dw >= 1 && sd <= (int64_t) UINT32_MAX &&
            sh <= (int64_t) UINT32_MAX && sw <= (int64_t) UINT32_MAX &&
            pd <= (int64_t) UINT32_MAX && ph <= (int64_t) UINT32_MAX &&
            pw <= (int64_t) UINT32_MAX && opd <= (int64_t) UINT32_MAX &&
            oph <= (int64_t) UINT32_MAX && opw <= (int64_t) UINT32_MAX &&
            dd <= (int64_t) UINT32_MAX && dh <= (int64_t) UINT32_MAX &&
            dw <= (int64_t) UINT32_MAX,
        "kuiper_convt3d: unsupported parameters or uint32 ABI overflow");

    const int64_t B = X.size(0), Cin = X.size(1), Din = X.size(2);
    const int64_t Hin = X.size(3), Win = X.size(4), WCin = W.size(0);
    const int64_t Cout = W.size(1), Kd = W.size(2);
    const int64_t Kh = W.size(3), Kw = W.size(4);
    TORCH_CHECK(
        B > 0 && Cin > 0 && Din > 0 && Hin > 0 && Win > 0 && Cout > 0 &&
            Kd > 0 && Kh > 0 && Kw > 0 && B <= (int64_t) UINT32_MAX &&
            Cin <= (int64_t) UINT32_MAX && Din <= (int64_t) UINT32_MAX &&
            Hin <= (int64_t) UINT32_MAX && Win <= (int64_t) UINT32_MAX &&
            Cout <= (int64_t) UINT32_MAX && Kd <= (int64_t) UINT32_MAX &&
            Kh <= (int64_t) UINT32_MAX && Kw <= (int64_t) UINT32_MAX &&
            Cin == WCin,
        "kuiper_convt3d: invalid shapes or uint32 ABI overflow");

    const bool has_bias = Bias_opt.has_value() && Bias_opt->defined();
    if (has_bias) {
        const auto &Bias = *Bias_opt;
        TORCH_CHECK(Bias.is_cuda() && Bias.device() == X.device() &&
                        Bias.scalar_type() == torch::kFloat32 &&
                        Bias.dim() == 1 && Bias.size(0) == Cout &&
                        Bias.is_contiguous(),
                    "kuiper_convt3d: invalid bias");
    }

    const c10::cuda::CUDAGuard device_guard(X.device());
    Kuiper_KB_ConvT3DGeneral_convt3d_raw_result r;
    if (has_bias) {
        r = Kuiper_KB_ConvT3DGeneral_convt3d_raw_alloc_bias_f32(
            (uint32_t) B, (uint32_t) Cin, (uint32_t) Din, (uint32_t) Hin,
            (uint32_t) Win, (uint32_t) Cout, (uint32_t) Kd, (uint32_t) Kh,
            (uint32_t) Kw, (uint32_t) sd, (uint32_t) sh, (uint32_t) sw,
            (uint32_t) pd, (uint32_t) ph, (uint32_t) pw, (uint32_t) opd,
            (uint32_t) oph, (uint32_t) opw, (uint32_t) dd, (uint32_t) dh,
            (uint32_t) dw, X.data_ptr<float>(), W.data_ptr<float>(),
            Bias_opt->data_ptr<float>());
    } else {
        r = Kuiper_KB_ConvT3DGeneral_convt3d_raw_alloc_zero_f32(
            (uint32_t) B, (uint32_t) Cin, (uint32_t) Din, (uint32_t) Hin,
            (uint32_t) Win, (uint32_t) Cout, (uint32_t) Kd, (uint32_t) Kh,
            (uint32_t) Kw, (uint32_t) sd, (uint32_t) sh, (uint32_t) sw,
            (uint32_t) pd, (uint32_t) ph, (uint32_t) pw, (uint32_t) opd,
            (uint32_t) oph, (uint32_t) opw, (uint32_t) dd, (uint32_t) dh,
            (uint32_t) dw, X.data_ptr<float>(), W.data_ptr<float>());
    }

    return torch::from_blob(
        r.snd.snd.snd,
        {B, Cout, (int64_t) r.fst, (int64_t) r.snd.fst,
         (int64_t) r.snd.snd.fst},
        [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_convt3d_general", &kuiper_convt3d_general_cuda,
          "Kuiper verified ConvTranspose3D forward");
}
