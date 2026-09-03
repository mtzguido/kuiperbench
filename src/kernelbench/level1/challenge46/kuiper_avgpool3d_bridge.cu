// KernelBench L1 #46: checks plus one extracted full-pipeline call.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>

#ifndef FStar_Pervasives_dfst
#define FStar_Pervasives_dfst(x) ((x).fst)
#endif
#ifndef FStar_Pervasives_dsnd
#define FStar_Pervasives_dsnd(x) ((x).snd)
#endif

#include "Kuiper_KB_AvgPool3D.h"
#include "Kuiper_KB_AvgPool3D.cu"

static constexpr __int128 U32_LIMIT = ((__int128) 1 << 32);
static constexpr __int128 KUIPER_MAX_NTHR = (__int128) 2097152 * 1024;

static __int128 pool_out(__int128 l, __int128 k, __int128 s, __int128 p)
{
    return (l + 2 * p - k) / s + 1;
}

static void check_contract(int64_t batch, int64_t channels,
                           int64_t depth, int64_t h, int64_t w,
                           int64_t k, int64_t s, int64_t p)
{
    const __int128 BC = (__int128) batch * channels;
    const __int128 D = depth, H = h, W = w, K = k, S = s, P = p;
    const __int128 padD = D + 2 * P, padH = H + 2 * P, padW = W + 2 * P;
    TORCH_CHECK(BC < U32_LIMIT && K < U32_LIMIT &&
                    padD < U32_LIMIT && padH < U32_LIMIT &&
                    padW < U32_LIMIT && K <= padD && K <= padH && K <= padW,
                "kuiper_avgpool3d: window is outside the verified range");
    const __int128 Do = pool_out(D, K, S, P), Ho = pool_out(H, K, S, P),
                   Wo = pool_out(W, K, S, P);
    auto fits = [](__int128 x) { return 0 <= x && x < U32_LIMIT; };
    auto launch = [](__int128 x) { return 0 < x && x <= KUIPER_MAX_NTHR; };
    TORCH_CHECK(fits(BC * D) && fits(BC * D * H) && fits(BC * D * H * W) &&
                    fits(BC * D * H * Wo) && fits(BC * D * Wo * H) &&
                    fits(BC * D * Wo * Ho) && fits(BC * (Ho * Wo) * D) &&
                    fits(BC * (Ho * Wo) * Do) && fits(Wo * S + K) &&
                    fits(Ho * S + K) && fits(Do * S + K) &&
                    launch(BC * D * H * Wo) && launch(BC * D * Wo * Ho) &&
                    launch(BC * (Ho * Wo) * Do),
                "kuiper_avgpool3d: shape exceeds verified u32/launch bounds");
}

torch::Tensor kuiper_avgpool3d_cuda(torch::Tensor X, int64_t kernel_size,
                                    int64_t stride, int64_t padding)
{
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32,
                "kuiper_avgpool3d: X must be CUDA float32");
    TORCH_CHECK(X.dim() == 5 && X.is_contiguous(),
                "kuiper_avgpool3d: X must be contiguous (B,C,D,H,W)");
    TORCH_CHECK(kernel_size >= 1 && stride >= 1 && padding >= 0 &&
                    (__int128) kernel_size < U32_LIMIT &&
                    (__int128) stride < U32_LIMIT &&
                    (__int128) padding < U32_LIMIT,
                "kuiper_avgpool3d: invalid pooling parameters");
    const int64_t B = X.size(0), C = X.size(1), D = X.size(2), H = X.size(3),
                  W = X.size(4);
    TORCH_CHECK(B > 0 && C > 0 && D > 0 && H > 0 && W > 0,
                "kuiper_avgpool3d: dimensions must be positive");
    check_contract(B, C, D, H, W, kernel_size, stride, padding);
    const c10::cuda::CUDAGuard device_guard(X.device());

    auto r = Kuiper_KB_AvgPool3D_avgpool3d_raw_alloc_f32(
        (uint32_t) kernel_size, (uint32_t) stride, (uint32_t) padding,
        (uint32_t) B, (uint32_t) C, (uint32_t) D, (uint32_t) H, (uint32_t) W,
        X.data_ptr<float>());
    return torch::from_blob(
        r.snd.snd.snd,
        {B, C, (int64_t) r.snd.snd.fst, (int64_t) r.snd.fst, (int64_t) r.fst},
        [](void *q) { cudaFree(q); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_avgpool3d", &kuiper_avgpool3d_cuda,
          "Kuiper verified AvgPool3D (single full-pipeline entry)");
}
