// Thin checked bridge for the fixed KernelBench L1 #75 configuration.
// Group selection, zero-bias construction, output allocation, and the full
// grouped ConvTranspose2D computation are inside the verified Kuiper entry.
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include "Kuiper_KB_ConvT2DGrouped75.h"
#include "Kuiper_KB_ConvT2DGrouped75.cu"

static torch::Tensor
kuiper_convt2d_grouped_cuda(torch::Tensor X, torch::Tensor W,
                            c10::optional<torch::Tensor> Bias_opt,
                            int64_t stride_h, int64_t stride_w, int64_t pad_h,
                            int64_t pad_w, int64_t out_pad_h, int64_t out_pad_w,
                            int64_t dil_h, int64_t dil_w, int64_t groups)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_convt2d_grouped: X and W must be CUDA");
    TORCH_CHECK(X.get_device() == W.get_device(),
                "kuiper_convt2d_grouped: X and W must be on the same device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32,
                "kuiper_convt2d_grouped: X and W must be float32");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous(),
                "kuiper_convt2d_grouped: X and W must be contiguous");
    TORCH_CHECK(X.sizes() == torch::IntArrayRef({16, 32, 128, 256}),
                "kuiper_convt2d_grouped: expected X shape (16,32,128,256)");
    TORCH_CHECK(W.sizes() == torch::IntArrayRef({32, 16, 3, 5}),
                "kuiper_convt2d_grouped: expected W shape (32,16,3,5)");
    TORCH_CHECK(!Bias_opt.has_value() || !Bias_opt->defined(),
                "kuiper_convt2d_grouped: challenge #75 requires bias=False");
    TORCH_CHECK(
        stride_h == 2 && stride_w == 3 && pad_h == 1 && pad_w == 2 &&
            out_pad_h == 0 && out_pad_w == 0 && dil_h == 2 && dil_w == 1 &&
            groups == 4,
        "kuiper_convt2d_grouped: parameters do not match challenge #75");

    at::cuda::CUDAGuard device_guard(X.device());
    float *out_ptr = Kuiper_KB_ConvT2DGrouped75_convt2d_grouped75_alloc_f32(
        X.data_ptr<float>(), W.data_ptr<float>());
    return torch::from_blob(
        out_ptr, {16, 64, 257, 766}, [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_convt2d_grouped", &kuiper_convt2d_grouped_cuda,
          "Kuiper verified grouped ConvTranspose2D for KernelBench L1 #75");
}
