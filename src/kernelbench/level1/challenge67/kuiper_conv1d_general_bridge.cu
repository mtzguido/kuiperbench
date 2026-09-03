#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_Conv1DAlloc.h"
#include "Kuiper_KB_Conv1DAlloc.cu"

static torch::Tensor kuiper_conv1d_general_cuda(
    torch::Tensor X, torch::Tensor W, c10::optional<torch::Tensor> Bias_opt,
    int64_t stride, int64_t pad, int64_t dilation, int64_t groups)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.device() == W.device(),
                "kuiper_conv1d: X and W must share a CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32,
                "kuiper_conv1d: X and W must be float32");
    TORCH_CHECK(X.dim() == 3 && W.dim() == 3 && X.is_contiguous() &&
                    W.is_contiguous(),
                "kuiper_conv1d: expected contiguous NCL/OCK tensors");
    TORCH_CHECK(groups == 1 && stride >= 1 && pad >= 0 && dilation >= 1 &&
                    stride <= (int64_t) UINT32_MAX &&
                    pad <= (int64_t) UINT32_MAX &&
                    dilation <= (int64_t) UINT32_MAX,
                "kuiper_conv1d: unsupported parameters or uint32 ABI overflow");

    const int64_t B = X.size(0), Cin = X.size(1), Lin = X.size(2);
    const int64_t Cout = W.size(0), WCin = W.size(1), K = W.size(2);
    TORCH_CHECK(B > 0 && Cin > 0 && Lin > 0 && Cout > 0 && K > 0 &&
                    B <= (int64_t) UINT32_MAX && Cin <= (int64_t) UINT32_MAX &&
                    Lin <= (int64_t) UINT32_MAX &&
                    Cout <= (int64_t) UINT32_MAX && K <= (int64_t) UINT32_MAX &&
                    Cin == WCin,
                "kuiper_conv1d: invalid shapes or uint32 ABI overflow");

    const bool has_bias = Bias_opt.has_value() && Bias_opt->defined();
    if (has_bias) {
        const auto &Bias = *Bias_opt;
        TORCH_CHECK(Bias.is_cuda() && Bias.device() == X.device() &&
                        Bias.scalar_type() == torch::kFloat32 &&
                        Bias.dim() == 1 && Bias.size(0) == Cout &&
                        Bias.is_contiguous(),
                    "kuiper_conv1d: invalid bias");
    }

    const c10::cuda::CUDAGuard device_guard(X.device());
    Prims_dtuple2__uint32_t__float_ r;
    if (has_bias) {
        r = Kuiper_KB_Conv1DAlloc_conv1d_raw_alloc_bias_f32(
            (uint32_t) B, (uint32_t) Cin, (uint32_t) Lin, (uint32_t) Cout,
            (uint32_t) K, (uint32_t) stride, (uint32_t) pad,
            (uint32_t) dilation, X.data_ptr<float>(), W.data_ptr<float>(),
            Bias_opt->data_ptr<float>());
    } else {
        r = Kuiper_KB_Conv1DAlloc_conv1d_raw_alloc_zero_f32(
            (uint32_t) B, (uint32_t) Cin, (uint32_t) Lin, (uint32_t) Cout,
            (uint32_t) K, (uint32_t) stride, (uint32_t) pad,
            (uint32_t) dilation, X.data_ptr<float>(), W.data_ptr<float>());
    }

    return torch::from_blob(
        r.snd, {B, Cout, (int64_t) r.fst}, [](void *p) { cudaFree(p); },
        X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_conv1d_general", &kuiper_conv1d_general_cuda,
          "Kuiper verified Conv1D forward");
}
