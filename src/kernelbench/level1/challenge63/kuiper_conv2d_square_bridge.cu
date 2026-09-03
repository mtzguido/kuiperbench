// Thin checked boundary for the fixed KernelBench L1 #63 operation.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_Conv2DSquare.h"
#include "Kuiper_KB_Conv2DSquare.cu"

static torch::Tensor kuiper_conv2d_square_cuda(torch::Tensor X, torch::Tensor W)
{
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && X.device() == W.device(),
                "kuiper_conv2d_square: X and W must share a CUDA device");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    W.scalar_type() == torch::kFloat32 &&
                    X.is_contiguous() && W.is_contiguous(),
                "kuiper_conv2d_square: X and W must be contiguous float32");
    TORCH_CHECK(X.sizes() == torch::IntArrayRef({16, 16, 1024, 1024}) &&
                    W.sizes() == torch::IntArrayRef({128, 16, 3, 3}),
                "kuiper_conv2d_square: unexpected input or weight shape");

    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_Conv2DSquare_conv2d_square63_alloc_f32(
        X.data_ptr<float>(), W.data_ptr<float>());
    return torch::from_blob(
        out, {16, 128, 1022, 1022}, [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_conv2d_square", &kuiper_conv2d_square_cuda,
          "Kuiper verified Conv2D for KernelBench L1 #63");
}
