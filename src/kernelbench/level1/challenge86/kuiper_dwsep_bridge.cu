// Thin checked boundary for the complete verified fixed #86 composition.
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_SeparableConv2D.h"
#include "Kuiper_KB_SeparableConv2D.cu"

static torch::Tensor kuiper_dwsep_cuda(
    torch::Tensor X, torch::Tensor Wdw, torch::Tensor Wpw,
    c10::optional<torch::Tensor> BiasDw_opt,
    c10::optional<torch::Tensor> BiasPw_opt,
    int64_t stride, int64_t pad, int64_t dilation)
{
    TORCH_CHECK(X.is_cuda() && Wdw.is_cuda() && Wpw.is_cuda() &&
                    X.device() == Wdw.device() && X.device() == Wpw.device(),
                "kuiper #86: tensors must be on the same CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    Wdw.scalar_type() == torch::kFloat32 &&
                    Wpw.scalar_type() == torch::kFloat32 &&
                    X.is_contiguous() && Wdw.is_contiguous() &&
                    Wpw.is_contiguous(),
                "kuiper #86: tensors must be contiguous float32");
    TORCH_CHECK(X.sizes() == torch::IntArrayRef({16, 64, 512, 512}) &&
                    Wdw.sizes() == torch::IntArrayRef({64, 1, 3, 3}) &&
                    Wpw.sizes() == torch::IntArrayRef({128, 64, 1, 1}),
                "kuiper #86: unexpected input or weight shape");
    TORCH_CHECK((!BiasDw_opt.has_value() || !BiasDw_opt->defined()) &&
                    (!BiasPw_opt.has_value() || !BiasPw_opt->defined()),
                "kuiper #86: bias must be disabled");
    TORCH_CHECK(stride == 1 && pad == 1 && dilation == 1,
                "kuiper #86: parameters do not match the verified challenge");

    const at::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_SeparableConv2D_separable86_alloc_f32(
        X.data_ptr<float>(), Wdw.data_ptr<float>(), Wpw.data_ptr<float>());
    return torch::from_blob(
        out, {16, 128, 512, 512}, [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_dwsep", &kuiper_dwsep_cuda,
          "Kuiper verified fixed KernelBench #86 separable Conv2D");
}
