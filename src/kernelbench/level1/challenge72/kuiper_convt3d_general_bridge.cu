// Thin checked bridge for the fixed KernelBench L1 #72 configuration.
// The complete grouped operation and all GPU allocation live in Kuiper.
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include "Kuiper_KB_ConvT3DGrouped72.h"
#include "Kuiper_KB_ConvT3DGrouped72.cu"

static torch::Tensor kuiper_convt3d_general_cuda(
    torch::Tensor X, torch::Tensor W, c10::optional<torch::Tensor> Bias_opt,
    int64_t stride_d, int64_t stride_h, int64_t stride_w, int64_t pad_d,
    int64_t pad_h, int64_t pad_w, int64_t out_pad_d, int64_t out_pad_h,
    int64_t out_pad_w, int64_t dil_d, int64_t dil_h, int64_t dil_w,
    int64_t groups)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_convt3d_general: X and W must be CUDA");
    TORCH_CHECK(X.get_device() == W.get_device(),
                "kuiper_convt3d_general: X and W must be on the same device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32,
                "kuiper_convt3d_general: X and W must be float32");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous(),
                "kuiper_convt3d_general: X and W must be contiguous");
    TORCH_CHECK(X.sizes() == torch::IntArrayRef({8, 32, 12, 24, 48}),
                "kuiper_convt3d_general: expected X shape (8,32,12,24,48)");
    TORCH_CHECK(W.sizes() == torch::IntArrayRef({32, 8, 3, 5, 7}),
                "kuiper_convt3d_general: expected W shape (32,8,3,5,7)");
    TORCH_CHECK(!Bias_opt.has_value() || !Bias_opt->defined(),
                "kuiper_convt3d_general: challenge #72 requires bias=False");
    TORCH_CHECK(
        stride_d == 2 && stride_h == 2 && stride_w == 2 && pad_d == 1 &&
            pad_h == 2 && pad_w == 3 && out_pad_d == 1 && out_pad_h == 1 &&
            out_pad_w == 1 && dil_d == 1 && dil_h == 1 && dil_w == 1 &&
            groups == 4,
        "kuiper_convt3d_general: parameters do not match challenge #72");

    at::cuda::CUDAGuard device_guard(X.device());
    float *out_ptr = Kuiper_KB_ConvT3DGrouped72_convt3d_grouped72_alloc_f32(
        X.data_ptr<float>(), W.data_ptr<float>());
    return torch::from_blob(
        out_ptr, {8, 32, 24, 48, 96}, [](void *p) { cudaFree(p); },
        X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_convt3d_general", &kuiper_convt3d_general_cuda,
          "Kuiper verified grouped ConvTranspose3D for KernelBench L1 #72");
}
