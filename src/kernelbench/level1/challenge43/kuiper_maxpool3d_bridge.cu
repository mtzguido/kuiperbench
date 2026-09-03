// KernelBench L1 #43: checks plus one extracted full-pipeline call.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>

#include "Kuiper_KB_MaxPool3D.h"
#include "Kuiper_KB_MaxPool3D.cu"

static constexpr __int128 U32_LIMIT = ((__int128) 1 << 32);
static constexpr __int128 KUIPER_MAX_NTHR = (__int128) 2097152 * 1024;

static __int128 pool_out(__int128 l, __int128 k, __int128 s, __int128 p,
                         __int128 d)
{
    return (l + 2 * p - (d * (k - 1) + 1)) / s + 1;
}

static void check_full_contract(int64_t batch, int64_t channels,
                                int64_t depth, int64_t h, int64_t w,
                                int64_t k, int64_t s, int64_t p, int64_t d)
{
    const __int128 BC = (__int128) batch * channels;
    const __int128 D = depth, H = h, W = w;
    const __int128 K = k, S = s, P = p, Dil = d;
    const __int128 span = Dil * (K - 1) + 1;
    const __int128 padD = D + 2 * P, padH = H + 2 * P, padW = W + 2 * P;
    TORCH_CHECK(BC < U32_LIMIT && span < U32_LIMIT &&
                    padD < U32_LIMIT && padH < U32_LIMIT &&
                    padW < U32_LIMIT && span <= padD && span <= padH &&
                    span <= padW,
                "kuiper_maxpool3d: window is outside the verified range");
    const __int128 Do = pool_out(D, K, S, P, Dil);
    const __int128 Ho = pool_out(H, K, S, P, Dil);
    const __int128 Wo = pool_out(W, K, S, P, Dil);
    auto fits = [](__int128 x) { return 0 <= x && x < U32_LIMIT; };
    auto launch = [](__int128 x) { return 0 < x && x <= KUIPER_MAX_NTHR; };
    TORCH_CHECK(fits(BC * D) && fits(BC * D * H) && fits(BC * D * H * W) &&
                    fits(BC * D * H * Wo) && fits(BC * D * Wo * H) &&
                    fits(BC * D * Wo * Ho) && fits(BC * (Ho * Wo) * D) &&
                    fits(BC * (Ho * Wo) * Do) && fits(Wo * S + K * Dil) &&
                    fits(Ho * S + K * Dil) && fits(Do * S + K * Dil) &&
                    launch(BC * D * H * Wo) && launch(BC * D * Wo * Ho) &&
                    launch(BC * (Ho * Wo) * Do),
                "kuiper_maxpool3d: shape exceeds verified u32/launch bounds");
}

torch::Tensor kuiper_maxpool3d_cuda(torch::Tensor X, int64_t kernel_size,
                                    int64_t stride, int64_t padding,
                                    int64_t dilation)
{
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32,
                "kuiper_maxpool3d: X must be CUDA float32");
    TORCH_CHECK(X.dim() == 5 && X.is_contiguous(),
                "kuiper_maxpool3d: X must be contiguous (B,C,D,H,W)");
    TORCH_CHECK(kernel_size >= 1 && stride >= 1 && padding >= 1 &&
                    dilation >= 1 && (__int128) kernel_size < U32_LIMIT &&
                    (__int128) stride < U32_LIMIT &&
                    (__int128) padding < U32_LIMIT &&
                    (__int128) dilation < U32_LIMIT,
                "kuiper_maxpool3d: invalid pooling parameters");
    const int64_t B = X.size(0), C = X.size(1), D = X.size(2);
    const int64_t H = X.size(3), W = X.size(4);
    TORCH_CHECK(B > 0 && C > 0 && D > 0 && H > 0 && W > 0,
                "kuiper_maxpool3d: dimensions must be positive");
    check_full_contract(B, C, D, H, W, kernel_size, stride, padding, dilation);
    const c10::cuda::CUDAGuard device_guard(X.device());

    auto r = Kuiper_KB_MaxPool3D_maxpool3d_raw_alloc_f32(
        (uint32_t) kernel_size, (uint32_t) stride,
        (uint32_t) padding, (uint32_t) dilation,
        (uint32_t) B, (uint32_t) C, (uint32_t) D, (uint32_t) H, (uint32_t) W,
        X.data_ptr<float>());

    return torch::from_blob(
        r.output,
        {B, C, (int64_t) r.d_out, (int64_t) r.h_out, (int64_t) r.w_out},
        [](void *q) { cudaFree(q); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_maxpool3d", &kuiper_maxpool3d_cuda,
          "Kuiper verified MaxPool3D (single full-pipeline entry)");
}
