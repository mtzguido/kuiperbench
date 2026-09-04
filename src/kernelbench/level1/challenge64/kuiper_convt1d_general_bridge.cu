#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_ConvT1DGeneral.h"
#include "Kuiper_KB_ConvT1DGeneral.cu"

static torch::Tensor kuiper_convt1d_general_cuda(
    torch::Tensor X, torch::Tensor W, c10::optional<torch::Tensor> Bias_opt,
    int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation,
    int64_t groups)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.device() == W.device(),
                "kuiper_convt1d: X and W must share a CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32,
                "kuiper_convt1d: X and W must be float32");
    TORCH_CHECK(X.dim() == 3 && W.dim() == 3 && X.is_contiguous() &&
                    W.is_contiguous(),
                "kuiper_convt1d: expected contiguous NCL/IOK tensors");
    TORCH_CHECK(!Bias_opt.has_value() || !Bias_opt->defined(),
                "kuiper_convt1d: these challenges require bias=False");
    TORCH_CHECK(
        groups == 1 && stride >= 1 && padding >= 0 && output_padding >= 0 &&
            dilation >= 1 && stride <= (int64_t) UINT32_MAX &&
            padding <= (int64_t) UINT32_MAX &&
            output_padding <= (int64_t) UINT32_MAX &&
            dilation <= (int64_t) UINT32_MAX,
        "kuiper_convt1d: unsupported parameters or uint32 ABI overflow");

    const int64_t B = X.size(0), Cin = X.size(1), Lin = X.size(2);
    const int64_t WCin = W.size(0), Cout = W.size(1), K = W.size(2);
    TORCH_CHECK(B > 0 && Cin > 0 && Lin > 0 && Cout > 0 && K > 0 &&
                    B <= (int64_t) UINT32_MAX && Cin <= (int64_t) UINT32_MAX &&
                    Lin <= (int64_t) UINT32_MAX &&
                    Cout <= (int64_t) UINT32_MAX && K <= (int64_t) UINT32_MAX &&
                    Cin == WCin,
                "kuiper_convt1d: invalid shapes or uint32 ABI overflow");

    const c10::cuda::CUDAGuard device_guard(X.device());
    const auto r = Kuiper_KB_ConvT1DGeneral_convt1d_general_alloc_f32(
        (uint32_t) B, (uint32_t) Cin, (uint32_t) Lin, (uint32_t) Cout,
        (uint32_t) K, (uint32_t) stride, (uint32_t) padding,
        (uint32_t) output_padding, (uint32_t) dilation, X.data_ptr<float>(),
        W.data_ptr<float>());

    return torch::from_blob(
        r.output, {B, Cout, (int64_t) r.l_out}, [](void *p) { cudaFree(p); },
        X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_convt1d_general", &kuiper_convt1d_general_cuda,
          "Kuiper verified ConvTranspose1D forward");
}
