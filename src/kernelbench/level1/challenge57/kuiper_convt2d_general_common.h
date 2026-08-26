// Common bridge body for KernelBench L1 #57/#65/#69/#71 — ConvTranspose2D
// forward (general): asymmetric inputs/kernels, stride, pad, dilation,
// optional bias.  Each per-challenge bridge `#define`s a unique
// TORCH_EXTENSION_NAME and `#include`s this header.
//
// The output-size math and output allocation now live INSIDE the
// verification boundary:
//   * Output dims [Hout]/[Wout] come from the VERIFIED extracted helper
//     `Kuiper_KB_ConvT2DGeneral_convt_out_dim` (no C++ formula).
//   * The output buffer is allocated INSIDE the verified self-allocating
//     entry `Kuiper_KB_ConvT2DGeneral_convt2d_general_alloc_f32`
//     (cudaMalloc via KPR_GPU_ALLOC) and returned as a bare device
//     pointer, which we wrap in a torch::Tensor with a cudaFree deleter
//     (no `torch::empty`).
// This driver therefore performs NO ConvTranspose output-size arithmetic
// and NO output allocation.  It only checks raw-dimension contracts and
// handles the bias scratch copy/zero.
#include <torch/extension.h>
#include "Kuiper_KB_ConvT2DGeneral.h"
#include "Kuiper_KB_ConvT2DGeneral.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

// Forward: y = convT2d(x, w) + bias  (bias optional; if undefined, zeroed scratch is used)
static torch::Tensor kuiper_convt2d_general_cuda(
        torch::Tensor X, torch::Tensor W,
        c10::optional<torch::Tensor> Bias_opt,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t out_pad_h, int64_t out_pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_convt2d_general: X and W must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                W.scalar_type() == torch::kFloat32,
                "kuiper_convt2d_general: X and W must be float32");
    TORCH_CHECK(X.dim() == 4 && W.dim() == 4,
                "kuiper_convt2d_general: X and W must be 4D");
    TORCH_CHECK(groups == 1,
                "kuiper_convt2d_general: only groups=1 supported");
    TORCH_CHECK(stride_h >= 1 && stride_w >= 1,
                "kuiper_convt2d_general: stride must be >= 1");
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0,
                "kuiper_convt2d_general: pad must be >= 0");
    TORCH_CHECK(out_pad_h >= 0 && out_pad_w >= 0,
                "kuiper_convt2d_general: output_padding must be >= 0");
    TORCH_CHECK(dil_h >= 1 && dil_w >= 1,
                "kuiper_convt2d_general: dilation must be >= 1");
    TORCH_CHECK(out_pad_h < stride_h && out_pad_w < stride_w,
                "kuiper_convt2d_general: output_padding must be < stride "
                "(PyTorch invariant)");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    int64_t B    = Xc.size(0);
    int64_t Cin  = Xc.size(1);
    int64_t Hin  = Xc.size(2);
    int64_t Win  = Xc.size(3);
    int64_t WCin = Wc.size(0);     // ConvT layout: (Cin, Cout, kH, kW)
    int64_t Cout = Wc.size(1);
    int64_t Kh   = Wc.size(2);
    int64_t Kw   = Wc.size(3);
    TORCH_CHECK(Cin == WCin, "kuiper_convt2d_general: Cin mismatch (X.size(1) != W.size(0))");
    TORCH_CHECK(B > 0 && Cin > 0 && Hin > 0 && Win > 0 && Cout > 0
                && Kh > 0 && Kw > 0,
                "kuiper_convt2d_general: shapes must be positive");

    // VERIFIED ConvTranspose2D output-size formula (extracted from
    // Kuiper.KB.ConvT2DGeneral), per axis:
    //   L_out = (L_in - 1)*S - 2*P + D*(K - 1) + output_padding + 1.
    // The non-negative part [pos = (L_in-1)*S + D*(K-1) + opad + 1] is
    // computed in int64 ONLY as an underflow/overflow guard (it is NOT the
    // output value): the [pos > 2*P] check discharges the verified helper's
    // [2*P <= pos] precondition (no size_t underflow, output >= 1) and the
    // [pos <= UINT32_MAX] check discharges its [SZ.fits] precondition.  The
    // actual output dimension is produced by the verified helper.
    int64_t pos_h = (Hin - 1) * stride_h + dil_h * (Kh - 1) + out_pad_h + 1;
    int64_t pos_w = (Win - 1) * stride_w + dil_w * (Kw - 1) + out_pad_w + 1;
    TORCH_CHECK(pos_h > 2 * pad_h && pos_w > 2 * pad_w,
                "kuiper_convt2d_general: zero-sized output");
    TORCH_CHECK(pos_h <= (int64_t)UINT32_MAX && pos_w <= (int64_t)UINT32_MAX,
                "kuiper_convt2d_general: output-size intermediate out of u32 range");
    int64_t Hout = (int64_t)Kuiper_KB_ConvT2DGeneral_convt_out_dim(
        (uint32_t)Hin, (uint32_t)stride_h, (uint32_t)dil_h, (uint32_t)Kh,
        (uint32_t)pad_h, (uint32_t)out_pad_h);
    int64_t Wout = (int64_t)Kuiper_KB_ConvT2DGeneral_convt_out_dim(
        (uint32_t)Win, (uint32_t)stride_w, (uint32_t)dil_w, (uint32_t)Kw,
        (uint32_t)pad_w, (uint32_t)out_pad_w);

    int64_t nthr = B * Cout * Hout * Wout;
    int64_t xnel = B * Cin  * Hin  * Win;
    int64_t wnel = Cin * Cout * Kh * Kw;
    int64_t ynel = nthr;
    TORCH_CHECK(B    <= (int64_t)UINT32_MAX &&
                Cin  <= (int64_t)UINT32_MAX &&
                Hin  <= (int64_t)UINT32_MAX &&
                Win  <= (int64_t)UINT32_MAX &&
                Cout <= (int64_t)UINT32_MAX &&
                Kh   <= (int64_t)UINT32_MAX &&
                Kw   <= (int64_t)UINT32_MAX &&
                Hout <= (int64_t)UINT32_MAX &&
                Wout <= (int64_t)UINT32_MAX &&
                xnel <= (int64_t)UINT32_MAX &&
                wnel <= (int64_t)UINT32_MAX &&
                ynel <= (int64_t)UINT32_MAX &&
                stride_h <= (int64_t)UINT32_MAX &&
                stride_w <= (int64_t)UINT32_MAX &&
                pad_h    <= (int64_t)UINT32_MAX &&
                pad_w    <= (int64_t)UINT32_MAX &&
                dil_h    <= (int64_t)UINT32_MAX &&
                dil_w    <= (int64_t)UINT32_MAX &&
                // SZ.fits side conditions
                (Kh * Kw)        <= (int64_t)UINT32_MAX &&
                (Cin * Kh * Kw)  <= (int64_t)UINT32_MAX &&
                (Hout * Wout)    <= (int64_t)UINT32_MAX &&
                (Cout * Hout * Wout) <= (int64_t)UINT32_MAX &&
                (Hout + pad_h)   <= (int64_t)UINT32_MAX &&
                (Wout + pad_w)   <= (int64_t)UINT32_MAX &&
                (Kh * dil_h)     <= (int64_t)UINT32_MAX &&
                (Kw * dil_w)     <= (int64_t)UINT32_MAX &&
                nthr <= KUIPER_MAX_NTHR,
                "kuiper_convt2d_general: shape out of range");

    // Bias scratch: a copy / zero-fill, not algorithmic math.
    float *bias_d = nullptr;
    bool bias_owned = false;
    size_t bb = (size_t)Cout * sizeof(float);
    if (Bias_opt.has_value() && Bias_opt->defined()) {
        auto Bias = Bias_opt->contiguous().to(X.device()).to(torch::kFloat32);
        TORCH_CHECK(Bias.numel() == Cout,
                    "kuiper_convt2d_general: bias.numel() != Cout");
        TORCH_CHECK(cudaMalloc(&bias_d, bb) == cudaSuccess,
                    "kuiper_convt2d_general: cudaMalloc bias scratch failed");
        TORCH_CHECK(cudaMemcpy(bias_d, Bias.data_ptr<float>(), bb,
                               cudaMemcpyDeviceToDevice) == cudaSuccess,
                    "kuiper_convt2d_general: cudaMemcpy bias scratch failed");
        bias_owned = true;
    } else {
        TORCH_CHECK(cudaMalloc(&bias_d, bb) == cudaSuccess,
                    "kuiper_convt2d_general: cudaMalloc bias failed");
        TORCH_CHECK(cudaMemset(bias_d, 0, bb) == cudaSuccess,
                    "kuiper_convt2d_general: cudaMemset bias failed");
        bias_owned = true;
    }

    // Self-allocating verified entry: allocates the (B*Cout*Hout*Wout) output
    // buffer inside the verification boundary, runs the verified kernel, and
    // returns the device pointer (ownership passes to us).
    float *out_ptr = Kuiper_KB_ConvT2DGeneral_convt2d_general_alloc_f32(
        (uint32_t)B, (uint32_t)Cin, (uint32_t)Hin, (uint32_t)Win,
        (uint32_t)Cout, (uint32_t)Kh, (uint32_t)Kw,
        (uint32_t)stride_h, (uint32_t)stride_w,
        (uint32_t)pad_h, (uint32_t)pad_w,
        (uint32_t)dil_h, (uint32_t)dil_w,
        (uint32_t)Hout, (uint32_t)Wout,
        Xc.data_ptr<float>(), Wc.data_ptr<float>(), bias_d);

    if (bias_owned) cudaFree(bias_d);

    // Wrap the Kuiper-allocated device buffer in a tensor that owns it: the
    // deleter cudaFree's it when the tensor is destroyed.
    auto Y = torch::from_blob(
        out_ptr, {B, Cout, Hout, Wout},
        [](void *p) { cudaFree(p); },
        Xc.options());
    return Y;
}
