// Common bridge body for KernelBench L1 #82..#85 — depthwise 2D conv
// forward (groups=in_channels, channel-multiplier 1, optional bias).
//
// Each per-challenge bridge `#include`s this file once, then provides
// its own PYBIND11_MODULE block.  All four challenges share one
// verified path through Karamel-extracted Kuiper.KB.DepthwiseConv2D.
//
// The two pieces of arithmetic/allocation that used to live in
// UNVERIFIED C++ now live INSIDE the verification boundary:
//   * Output dims [Hout]/[Wout] come from the VERIFIED extracted helper
//     `Kuiper_KB_DepthwiseConv2D_dwconv2d_out_dim` (no C++ division).
//   * The output buffer is allocated INSIDE the verified self-allocating
//     entry `Kuiper_KB_DepthwiseConv2D_dwconv2d_alloc_f32` (cudaMalloc via
//     KPR_GPU_ALLOC) and returned as a bare device pointer, which we wrap
//     in a torch::Tensor with a cudaFree deleter (no `torch::empty`).
// This driver therefore performs NO conv output-size arithmetic and NO
// output allocation.  It only checks raw-dimension contracts and handles
// the bias scratch copy/zero (a copy/zero, not algorithmic math).
//
// PyTorch reference:
//   nn.Conv2d(C, C, (kH, kW), stride, padding, dilation=1, groups=C, bias=...)
//   X: (B, C, H_in, W_in)
//   W: (C, 1, kH, kW)            (depthwise weight layout)
//   Y: (B, C, H_out, W_out)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_DepthwiseConv2D.h"
#include "Kuiper_KB_DepthwiseConv2D.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

static torch::Tensor kuiper_dwconv2d_cuda(
        torch::Tensor X, torch::Tensor W,
        c10::optional<torch::Tensor> Bias_opt,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_dwconv2d: X and W must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                W.scalar_type() == torch::kFloat32,
                "kuiper_dwconv2d: X and W must be float32");
    TORCH_CHECK(X.dim() == 4 && W.dim() == 4,
                "kuiper_dwconv2d: X and W must be 4D");
    TORCH_CHECK(stride_h == stride_w,
                "kuiper_dwconv2d: stride_h must equal stride_w");
    TORCH_CHECK(pad_h == pad_w,
                "kuiper_dwconv2d: pad_h must equal pad_w");
    TORCH_CHECK(dil_h == 1 && dil_w == 1,
                "kuiper_dwconv2d: only dilation=1 supported");
    TORCH_CHECK(stride_h >= 1, "kuiper_dwconv2d: stride must be >= 1");
    TORCH_CHECK(pad_h >= 0,    "kuiper_dwconv2d: pad must be >= 0");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    int64_t B    = Xc.size(0);
    int64_t Cin  = Xc.size(1);
    int64_t Hin  = Xc.size(2);
    int64_t Win  = Xc.size(3);
    int64_t Cw   = Wc.size(0);
    int64_t Cmul = Wc.size(1);
    int64_t Kh   = Wc.size(2);
    int64_t Kw   = Wc.size(3);
    TORCH_CHECK(Cin == Cw,
                "kuiper_dwconv2d: weight C dim must equal input C dim");
    TORCH_CHECK(Cmul == 1,
                "kuiper_dwconv2d: only channel-multiplier 1 supported "
                "(weight shape (C, 1, kH, kW))");
    TORCH_CHECK(groups == Cin,
                "kuiper_dwconv2d: only groups == in_channels supported");
    TORCH_CHECK(B > 0 && Cin > 0 && Hin > 0 && Win > 0
                && Kh > 0 && Kw > 0,
                "kuiper_dwconv2d: shapes must be positive");

    // Padded input must be at least as large as the kernel.  This discharges
    // the `k <= n + 2*pad` precondition of the verified `dwconv2d_out_dim`
    // (no size_t underflow) AND keeps the padded extents in u32 so the helper
    // does not overflow internally.
    int64_t Hpad = Hin + 2 * pad_h;
    int64_t Wpad = Win + 2 * pad_w;
    TORCH_CHECK(Hpad >= Kh && Wpad >= Kw,
                "kuiper_dwconv2d: padded input smaller than kernel");
    TORCH_CHECK(Hpad <= (int64_t)UINT32_MAX && Wpad <= (int64_t)UINT32_MAX,
                "kuiper_dwconv2d: padded input out of u32 range");

    // VERIFIED depthwise-conv output-size division (extracted from
    // Kuiper.KB.DepthwiseConv2D):
    //   Hout = (Hin + 2*pad_h - Kh) / stride_h + 1
    //   Wout = (Win + 2*pad_w - Kw) / stride_w + 1
    // No hand-written C++ division here.
    int64_t Hout = (int64_t)Kuiper_KB_DepthwiseConv2D_dwconv2d_out_dim(
        (uint32_t)Hin, (uint32_t)Kh, (uint32_t)stride_h, (uint32_t)pad_h);
    int64_t Wout = (int64_t)Kuiper_KB_DepthwiseConv2D_dwconv2d_out_dim(
        (uint32_t)Win, (uint32_t)Kw, (uint32_t)stride_w, (uint32_t)pad_w);
    TORCH_CHECK(Hout >= 1 && Wout >= 1,
                "kuiper_dwconv2d: zero-sized output");

    int64_t nthr = B * Cin * Hout * Wout;
    int64_t xnel = B * Cin * Hin * Win;
    int64_t wnel = Cin * 1 * Kh * Kw;
    int64_t ynel = nthr;
    // Raw-dimension contracts discharging the verified `dwconv2d_size_req`
    // precondition (all multiplications fit u32; nthr within launch bound).
    TORCH_CHECK(B    <= (int64_t)UINT32_MAX &&
                Cin  <= (int64_t)UINT32_MAX &&
                Hin  <= (int64_t)UINT32_MAX &&
                Win  <= (int64_t)UINT32_MAX &&
                Kh   <= (int64_t)UINT32_MAX &&
                Kw   <= (int64_t)UINT32_MAX &&
                Hout <= (int64_t)UINT32_MAX &&
                Wout <= (int64_t)UINT32_MAX &&
                xnel <= (int64_t)UINT32_MAX &&
                wnel <= (int64_t)UINT32_MAX &&
                ynel <= (int64_t)UINT32_MAX &&
                stride_h <= (int64_t)UINT32_MAX &&
                pad_h    <= (int64_t)UINT32_MAX &&
                // SZ.fits side conditions (h_out*stride+kh, w_out*stride+kw)
                (Hout*stride_h + Kh) <= (int64_t)UINT32_MAX &&
                (Wout*stride_w + Kw) <= (int64_t)UINT32_MAX &&
                nthr <= KUIPER_MAX_NTHR,
                "kuiper_dwconv2d: shape out of range");

    // Bias: forward caller's tensor if defined, otherwise zeroed scratch.
    // This is a copy / zero-fill, not algorithmic math.
    float *bias_d = nullptr;
    bool bias_owned = false;
    if (Bias_opt.has_value() && Bias_opt->defined()) {
        auto Bias = Bias_opt->contiguous().to(X.device()).to(torch::kFloat32);
        TORCH_CHECK(Bias.numel() == Cin,
                    "kuiper_dwconv2d: bias.numel() != C");
        size_t bb = (size_t)Cin * sizeof(float);
        float *scratch = nullptr;
        TORCH_CHECK(cudaMalloc(&scratch, bb) == cudaSuccess,
                    "kuiper_dwconv2d: cudaMalloc bias scratch failed");
        TORCH_CHECK(cudaMemcpy(scratch, Bias.data_ptr<float>(), bb,
                               cudaMemcpyDeviceToDevice) == cudaSuccess,
                    "kuiper_dwconv2d: cudaMemcpy bias scratch failed");
        bias_d = scratch;
        bias_owned = true;
    } else {
        size_t bb = (size_t)Cin * sizeof(float);
        TORCH_CHECK(cudaMalloc(&bias_d, bb) == cudaSuccess,
                    "kuiper_dwconv2d: cudaMalloc bias failed");
        TORCH_CHECK(cudaMemset(bias_d, 0, bb) == cudaSuccess,
                    "kuiper_dwconv2d: cudaMemset bias failed");
        bias_owned = true;
    }

    // Self-allocating verified entry: allocates the (B*Cin*Hout*Wout) output
    // buffer inside the verification boundary, runs the verified kernel, and
    // returns the device pointer (ownership passes to us).
    float *out_ptr = Kuiper_KB_DepthwiseConv2D_dwconv2d_alloc_f32(
        (uint32_t)B, (uint32_t)Cin, (uint32_t)Hin, (uint32_t)Win,
        (uint32_t)Kh, (uint32_t)Kw,
        (uint32_t)stride_h, (uint32_t)pad_h,
        (uint32_t)Hout, (uint32_t)Wout,
        Xc.data_ptr<float>(), Wc.data_ptr<float>(), bias_d);

    if (bias_owned) cudaFree(bias_d);

    // Wrap the Kuiper-allocated device buffer in a tensor that owns it: the
    // deleter cudaFree's it when the tensor is destroyed.
    auto Y = torch::from_blob(
        out_ptr, {B, Cin, Hout, Wout},
        [](void *p) { cudaFree(p); },
        Xc.options());
    return Y;
}
