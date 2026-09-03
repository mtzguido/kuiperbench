// Bridge for KernelBench L1 #44: AvgPool1D.
//
// PyTorch's nn.AvgPool1d(K, S, P) takes input (B, C, L) and produces
// (B, C, L_out).  Default count_include_pad=True so the divisor is K
// regardless of how much of the window lies in padding.  Padding
// contributes 0 to the sum.
//
// The verified, self-allocating entry point
// [Kuiper_KB_AvgPool1D_avgpool1d_raw_alloc_f32] takes ONLY the canonical
// (K,S,P,B,C,L) arguments and the input buffer, and *inside the verification
// boundary*:
//   * computes L_out via the verified [pool_out_len_1d_sz],
//   * allocates the (B*C, L_out) GPU output buffer (extracts to cudaMalloc),
//   * derives bc = B*C, supplies the implicit unit dilation, and runs
//     windowreduce with the f32 fadd
//     monoid (rid = 0, rop = +) writing the per-window SUM,
//   * divides every output element by K in place via the verified
//     Kuiper.KB.ScalarMul kernel (scaling by inv_k = 1/K), and
//   * returns the pair (L_out, output_device_ptr).
// Ownership of the returned buffer passes to this bridge, which wraps it in a
// torch::Tensor with a cudaFree deleter.  This driver therefore performs NO
// arithmetic, NO allocation, and NO second kernel launch; it only checks
// dimension contracts on the raw dims (the verified preconditions).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "Kuiper_KB_AvgPool1D.h"
#include "Kuiper_KB_AvgPool1D.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_avgpool1d_cuda(torch::Tensor X,
                                    int64_t kernel_size,
                                    int64_t stride,
                                    int64_t padding) {
    TORCH_CHECK(X.is_cuda(), "kuiper_avgpool1d: X must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
                "kuiper_avgpool1d: X must be float32");
    TORCH_CHECK(X.dim() == 3 && X.is_contiguous(),
                "kuiper_avgpool1d: X must be contiguous (B, C, L)");
    TORCH_CHECK(kernel_size >= 1 && stride >= 1 && padding >= 1,
                "kuiper_avgpool1d: k/s/p must be >= 1 (verified szp range)");

    int64_t B = X.size(0);
    int64_t C = X.size(1);
    int64_t L = X.size(2);
    // Raw-dimension contract checks discharging the verified preconditions of
    // [avgpool1d_alloc_f32].  No L_out is computed here: the kernel computes,
    // allocates, fills, scales, and returns it.  [kspan = D*(K-1)+1],
    // [padded = L+2P].
    const __int128 BC = (__int128) B * C;
    const __int128 kspan = kernel_size;
    const __int128 padded = (__int128) L + 2 * padding;
    const __int128 U = (__int128) UINT32_MAX;
    TORCH_CHECK(B > 0 && C > 0 && L > 0
                && BC > 0 && BC <= U
                && L <= U
                && kspan <= U
                && padded <= U
                && kspan <= padded                       /* window fits => L_out > 0 */
                && padded * stride + kernel_size <= U
                && BC * L <= U
                && BC * padded <= U
                && BC * padded <= KUIPER_MAX_NTHR,
                "kuiper_avgpool1d: shape out of verified u32 / launch range");

    const c10::cuda::CUDAGuard device_guard(X.device());
    Prims_dtuple2__uint32_t__float_ r =
        Kuiper_KB_AvgPool1D_avgpool1d_raw_alloc_f32(
            (uint32_t)kernel_size, (uint32_t)stride, (uint32_t)padding,
            (uint32_t)B, (uint32_t)C, (uint32_t)L,
            X.data_ptr<float>());

    int64_t L_out = (int64_t)r.fst;
    float *out_ptr = r.snd;

    // Wrap the Kuiper-allocated (cudaMalloc'd) device buffer in a tensor that
    // owns it: the deleter cudaFree's it when the tensor is destroyed.
    auto Y = torch::from_blob(
        out_ptr, {B, C, L_out},
        [](void *p) { cudaFree(p); },
        X.options());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_avgpool1d", &kuiper_avgpool1d_cuda,
          "Kuiper verified AvgPool1D (self-allocating: windowreduce fadd sum + verified ScalarMul /K)");
}
