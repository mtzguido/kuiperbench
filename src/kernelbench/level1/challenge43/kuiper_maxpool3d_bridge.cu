// Bridge for KernelBench L1 #43: MaxPool3D.
//
// PyTorch's nn.MaxPool3d(K, S, P, D) takes (B, C, D, H, W) and produces
// (B, C, D_out, H_out, W_out) with each *_out = floor((dim + 2P - D*(K-1) - 1)/S) + 1.
// Padding contributes implicit -inf to the max.
//
// MaxPool is separable: max{x[d+dd, h+dh, w+dw]} = max_dd max_dh max_dw x[..].
// We compose three passes of the verified, self-allocating entry
// [Kuiper_KB_MaxPool3D_maxpool3d_axis_alloc_f32].  Each pass takes ONLY raw
// (bc, L, K, S, P, D) and the input buffer, and *inside the verification
// boundary*: computes L_out, allocates the (bc, L_out) GPU output buffer
// (cudaMalloc), runs windowreduce with the f32 fmax monoid, and returns the
// pair (L_out, output_device_ptr).  This bridge does NO arithmetic that feeds
// the kernel, NO output allocation, and NO chunking loop; it only checks
// dimension contracts and performs the inter-pass permutes (PyTorch's, NOT
// verified by Kuiper):
//
//   pass 1: view input as (B*C*D*H, W);          reduce over W ⇒ (B,C,D,H,W_out)
//   permute (0,1,2,4,3): (B,C,D,W_out,H)
//   pass 2: view as (B*C*D*W_out, H);            reduce over H ⇒ (B,C,D,W_out,H_out)
//   permute (0,1,3,4,2): (B,C,W_out,H_out,D)
//   pass 3: view as (B*C*W_out*H_out, D);        reduce over D ⇒ (B,C,W_out,H_out,D_out)
//   permute (0,1,4,3,2) back to (B,C,D_out,H_out,W_out)
//
// See skeptic.txt for the exact verification gap (the three permute+contiguous
// copies).
#include <torch/extension.h>
#include <cuda_runtime.h>

// Karamel emits the verified entry's return value as a Prims dtuple2 struct,
// but the build drops the Prims namespace (-drop Prims), so the struct's
// definition is not present in the generated header.  Declare the ABI type
// here (matching Karamel's layout: { .fst = L_out, .snd = device ptr }) before
// including the extracted code.  Pure ABI declaration -- no kernel logic.
typedef struct Prims_dtuple2__uint32_t__float__s {
    uint32_t fst;
    float *snd;
} Prims_dtuple2__uint32_t__float_;

// Karamel's compound-literal compatibility macro (emitted by the gpu-branch
// krml but absent from this karamel runtime header).
#ifndef KRML_CLITERAL
#  if defined(__cplusplus)
#    define KRML_CLITERAL(x) x
#  else
#    define KRML_CLITERAL(x) (x)
#  endif
#endif

#include "Kuiper_KB_MaxPool3D.h"
#include "Kuiper_KB_MaxPool3D.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

// Raw-dimension contract checks discharging the verified preconditions of
// [maxpool3d_axis_alloc_f32] for one axis.  No L_out is computed here: the
// kernel computes, allocates, fills, and returns it.
static void check_axis_raw(int64_t bc, int64_t L,
                           int64_t k, int64_t s, int64_t p, int64_t d) {
    int64_t kspan  = d * (k - 1) + 1;
    int64_t padded = L + 2 * p;
    TORCH_CHECK(bc > 0 && bc <= (int64_t)UINT32_MAX
                && L <= (int64_t)UINT32_MAX
                && kspan <= (int64_t)UINT32_MAX
                && padded <= (int64_t)UINT32_MAX
                && kspan <= padded                       /* window fits => L_out > 0 */
                && padded * s + k * d <= (int64_t)UINT32_MAX
                && bc * L <= (int64_t)UINT32_MAX
                && bc * padded <= (int64_t)UINT32_MAX
                && bc * padded <= KUIPER_MAX_NTHR,
                "kuiper_maxpool3d: shape out of verified u32 / launch range");
}

// Run one self-allocating axis pass: returns the result as a tensor of the
// given shape (last dim = computed L_out) that owns the Kuiper-allocated
// (cudaMalloc'd) device buffer.
static torch::Tensor run_axis(torch::Tensor in, int64_t bc, int64_t L,
                              int64_t k, int64_t s, int64_t p, int64_t d,
                              std::vector<int64_t> out_shape) {
    check_axis_raw(bc, L, k, s, p, d);
    Prims_dtuple2__uint32_t__float_ r =
        Kuiper_KB_MaxPool3D_maxpool3d_axis_alloc_f32(
            (uint32_t)k, (uint32_t)s, (uint32_t)p, (uint32_t)d,
            (uint32_t)bc, (uint32_t)L, in.data_ptr<float>());
    int64_t L_out = (int64_t)r.fst;
    out_shape.back() = L_out;
    return torch::from_blob(
        r.snd, out_shape, [](void *q) { cudaFree(q); }, in.options());
}

torch::Tensor kuiper_maxpool3d_cuda(torch::Tensor X,
                                    int64_t kernel_size,
                                    int64_t stride,
                                    int64_t padding,
                                    int64_t dilation) {
    TORCH_CHECK(X.is_cuda(), "kuiper_maxpool3d: X must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
                "kuiper_maxpool3d: X must be float32");
    TORCH_CHECK(X.dim() == 5, "kuiper_maxpool3d: X must be (B, C, D, H, W)");
    TORCH_CHECK(kernel_size >= 1 && stride >= 1 && padding >= 1
                && dilation >= 1,
                "kuiper_maxpool3d: k/s/p/d must be >= 1 (verified szp range)");

    auto Xc = X.contiguous();
    int64_t B = Xc.size(0);
    int64_t C = Xc.size(1);
    int64_t D = Xc.size(2);
    int64_t H = Xc.size(3);
    int64_t W = Xc.size(4);

    TORCH_CHECK(B > 0 && C > 0 && D > 0 && H > 0 && W > 0,
                "kuiper_maxpool3d: input dimensions must be positive");
    // ── Pass 1: reduce over W axis. View as (B*C*D*H, W). ────────────
    auto Y1 = run_axis(Xc, B * C * D * H, W,
                       kernel_size, stride, padding, dilation,
                       {B, C, D, H, /*W_out*/ 0});          // (B,C,D,H,W_out)
    int64_t W_out = Y1.size(4);

    // ── Pass 2: reduce over H axis.  Permute (B,C,D,H,W_out) →
    // (B,C,D,W_out,H), make contiguous, view as (B*C*D*W_out, H). ────
    auto Y1p = Y1.permute({0, 1, 2, 4, 3}).contiguous();    // (B,C,D,W_out,H)
    auto Y2 = run_axis(Y1p, B * C * D * W_out, H,
                       kernel_size, stride, padding, dilation,
                       {B, C, D, W_out, /*H_out*/ 0});       // (B,C,D,W_out,H_out)
    int64_t H_out = Y2.size(4);

    // ── Pass 3: reduce over D axis.  Permute (B,C,D,W_out,H_out) →
    // (B,C,W_out,H_out,D), view as (B*C*W_out*H_out, D). ─────────────
    auto Y2p = Y2.permute({0, 1, 3, 4, 2}).contiguous();    // (B,C,W_out,H_out,D)
    auto Y3 = run_axis(Y2p, B * C * W_out * H_out, D,
                       kernel_size, stride, padding, dilation,
                       {B, C, W_out, H_out, /*D_out*/ 0});   // (B,C,W_out,H_out,D_out)

    // Permute (B,C,W_out,H_out,D_out) → (B,C,D_out,H_out,W_out).
    return Y3.permute({0, 1, 4, 3, 2}).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_maxpool3d", &kuiper_maxpool3d_cuda,
          "Kuiper verified MaxPool3D (three self-allocating windowreduce passes)");
}
