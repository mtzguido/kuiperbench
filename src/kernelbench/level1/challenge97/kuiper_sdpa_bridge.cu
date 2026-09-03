// Bridge for KernelBench L1 #97: Scaled Dot-Product Attention.
//
// Reference:
//   F.scaled_dot_product_attention(Q, K, V)
//     = softmax(Q K^T / sqrt(d_k)) V        (no mask, no dropout)
//
// Q, K, V have shape (B, H, S, D). The bridge passes those four raw
// dimensions to `Kuiper_KB_SDPA_sdpa_f32`; Kuiper proves the internal
// (B*H,S,D) view and then chains four verified primitives (BatchedGEMM,
// ScalarMul, RowSoftmax, BatchedGEMM) under a rank-4 functional spec.
//
// The bridge validates the tensor ABI and calls one self-allocating Kuiper
// entry point. Kuiper owns the page-transpose view, scratch allocation, four
// verified stages, and their end-to-end functional specification.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "kuiper.h"
// Kuiper_KB_SDPA bundles ScalarMul + RowSoftmax internally (as static fns) and
// calls Kuiper_KB_BatchedGEMM_batched_gemm_f32 as an extern. We therefore
// include the SDPA orchestrator plus the BatchedGEMM kernel it depends on, in
// the global namespace. The two .cu files define disjoint static __hoisted_N
// kernels, so there is no symbol clash in this single translation unit.
#include "Kuiper_KB_BatchedGEMM.h"
#include "Kuiper_KB_BatchedGEMM.cu"
#include "Kuiper_KB_SDPA.h"
#include "Kuiper_KB_SDPA.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS = (int64_t) 2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t) 1024;

torch::Tensor kuiper_sdpa_cuda(torch::Tensor Q, torch::Tensor K,
                               torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() &&
                    Q.scalar_type() == torch::kFloat32 &&
                    K.scalar_type() == torch::kFloat32 &&
                    V.scalar_type() == torch::kFloat32 && Q.dim() == 4 &&
                    K.dim() == 4 && V.dim() == 4,
                "kuiper_sdpa: expected three 4-D float32 CUDA tensors");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "kuiper_sdpa: Q, K, V must share shape (B,H,S,D)");
    TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                "kuiper_sdpa: Q, K, V must be on the same CUDA device");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "kuiper_sdpa: Q, K, V must be contiguous");

    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);

    TORCH_CHECK(B > 0 && H > 0 && S > 0 && D > 0,
                "kuiper_sdpa: shape dims must be positive");
    using wide = unsigned __int128;
    const wide bh_w = (wide) B * (wide) H;
    TORCH_CHECK(bh_w <= (wide) UINT32_MAX && (wide) S <= (wide) UINT32_MAX &&
                    (wide) D <= (wide) UINT32_MAX,
                "kuiper_sdpa: dimensions exceed the uint32 ABI");
    const wide bhs = bh_w * (wide) S;
    const wide scores = bhs * (wide) S;
    const wide output = bhs * (wide) D;
    const wide launch_limit =
        (wide) KUIPER_MAX_BLOCKS * (wide) KUIPER_MAX_THREADS;
    TORCH_CHECK(scores <= launch_limit && output <= launch_limit &&
                    (wide) S * (wide) S <= launch_limit &&
                    bhs <= (wide) KUIPER_MAX_BLOCKS &&
                    (wide) S + (wide) KUIPER_MAX_THREADS <= (wide) UINT32_MAX,
                "kuiper_sdpa: shape out of supported range");

    const c10::cuda::CUDAGuard device_guard(Q.device());
    float *out = Kuiper_KB_SDPA_sdpa_f32(
        (uint32_t) B, (uint32_t) H, (uint32_t) S, (uint32_t) D,
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>());
    return torch::from_blob(
        out, {B, H, S, D}, [](void *p) { cudaFree(p); }, Q.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kuiper_sdpa", &kuiper_sdpa_cuda,
          "Verified Scaled Dot-Product Attention (single Kuiper_KB_SDPA "
          "orchestrator: BatchedGEMM + ScalarMul + RowSoftmax + BatchedGEMM)");
}
