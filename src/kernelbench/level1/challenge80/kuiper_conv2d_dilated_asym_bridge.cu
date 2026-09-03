// Thin checked boundary for the complete verified fixed #80 operation.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_Conv2DDilatedAsym.h"
#include "Kuiper_KB_Conv2DDilatedAsym.cu"

static torch::Tensor kuiper_conv2d_dilated_asym_cuda(
    torch::Tensor X, torch::Tensor W,
    c10::optional<torch::Tensor> Bias_opt,
    int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w, int64_t groups)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.device() == W.device(),
                "kuiper #80: inputs must be on the same CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32 &&
                    X.is_contiguous() && W.is_contiguous(),
                "kuiper #80: inputs must be contiguous float32 tensors");
    TORCH_CHECK(X.sizes() == torch::IntArrayRef({8, 32, 512, 512}) &&
                    W.sizes() == torch::IntArrayRef({64, 32, 5, 9}),
                "kuiper #80: unexpected input or weight shape");
    TORCH_CHECK(!Bias_opt.has_value() || !Bias_opt->defined(),
                "kuiper #80: bias must be disabled");
    TORCH_CHECK(stride_h == 1 && stride_w == 1 && pad_h == 2 && pad_w == 4 &&
                    dil_h == 2 && dil_w == 3 && groups == 1,
                "kuiper #80: parameters do not match the verified challenge");

    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out =
        Kuiper_KB_Conv2DDilatedAsym_conv2d_dilated_asym80_alloc_f32(
            X.data_ptr<float>(), W.data_ptr<float>());
    return torch::from_blob(
        out, {8, 64, 508, 496}, [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_conv2d_dilated_asym", &kuiper_conv2d_dilated_asym_cuda,
          "Kuiper verified fixed KernelBench #80 Conv2D");
}
