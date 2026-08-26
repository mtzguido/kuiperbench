// Bridge for KernelBench L1 #97: Scaled Dot-Product Attention.
//
// Reference:
//   F.scaled_dot_product_attention(Q, K, V)
//     = softmax(Q K^T / sqrt(d_k)) V        (no mask, no dropout)
//
// Q, K, V have shape (B, H, S, D). We flatten the batch+head dims to
// (B*H, S, D) and call the single verified Kuiper orchestrator
// `Kuiper_KB_SDPA_sdpa_f32`, which internally chains four verified
// primitives (BatchedGEMM, ScalarMul, RowSoftmax, BatchedGEMM) and carries
// a functional correctness proof for the whole composition.
//
// The bridge itself only does shape arithmetic, device allocation and the
// host-side K-transpose (a PyTorch primitive, not a Kuiper kernel — same
// pattern as L1 #17 "A B^T"). Every GPU computation goes through the
// Karamel-extracted, F*-verified entry point.

#include <torch/extension.h>
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

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_sdpa_cuda(torch::Tensor Q,
                               torch::Tensor K,
                               torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda()
                && Q.scalar_type() == torch::kFloat32
                && K.scalar_type() == torch::kFloat32
                && V.scalar_type() == torch::kFloat32
                && Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "kuiper_sdpa: expected three 4-D float32 CUDA tensors");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "kuiper_sdpa: Q, K, V must share shape (B,H,S,D)");

    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    int64_t BH = B * H;

    TORCH_CHECK(B > 0 && H > 0 && S > 0 && D > 0,
                "kuiper_sdpa: shape dims must be positive");
    TORCH_CHECK(BH * S * S <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && BH * S * D <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && S * S <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && S * D <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && BH * S <= KUIPER_MAX_BLOCKS
                && (S + KUIPER_MAX_THREADS) <= (int64_t)UINT32_MAX
                && BH <= (int64_t)UINT32_MAX
                && S  <= (int64_t)UINT32_MAX
                && D  <= (int64_t)UINT32_MAX,
                "kuiper_sdpa: shape out of supported range");

    // Flatten (B,H,S,D) -> (BH,S,D).
    auto Q3 = Q.contiguous().view({BH, S, D});
    auto V3 = V.contiguous().view({BH, S, D});
    // K^T per batch: (BH, D, S). Materialised contiguously host-side.
    auto KT = K.contiguous().view({BH, S, D}).transpose(-2, -1).contiguous();

    // Scratch (BH,S,S) scores and (BH,S,D) output, allocated for the
    // orchestrator to fill.
    float *gScores = nullptr;
    float *gOut = nullptr;
    TORCH_CHECK(cudaMalloc(&gScores, BH * S * S * sizeof(float)) == cudaSuccess,
                "kuiper_sdpa: scores alloc failed");
    TORCH_CHECK(cudaMalloc(&gOut, BH * S * D * sizeof(float)) == cudaSuccess,
                "kuiper_sdpa: out alloc failed");

    float scale = Kuiper_KB_SDPA_sdpa_scale_f32((uint32_t)D);

    // Single verified entry point:
    //   out = softmax((Q · K^T) * scale) · V
    Kuiper_KB_SDPA_sdpa_f32(
        (uint32_t)BH, (uint32_t)S, (uint32_t)D, scale,
        Q3.data_ptr<float>(), KT.data_ptr<float>(), V3.data_ptr<float>(),
        gScores, gOut);

    auto out = torch::empty({B, H, S, D}, Q.options());
    cudaMemcpy(out.data_ptr<float>(), gOut,
               BH * S * D * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaFree(gScores);
    cudaFree(gOut);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_sdpa", &kuiper_sdpa_cuda,
          "Verified Scaled Dot-Product Attention (single Kuiper_KB_SDPA "
          "orchestrator: BatchedGEMM + ScalarMul + RowSoftmax + BatchedGEMM)");
}
