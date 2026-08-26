// Bridge for KernelBench L1 #80: 2D conv forward with square input,
// asymmetric kernel, asymmetric padding, asymmetric dilation,
// stride=1 (single).  Wraps the verified
// [Kuiper_KB_Conv2DDilatedAsym_conv2d_dilated_asym_f32] entry point.
#include <torch/extension.h>
#include "Kuiper_KB_Conv2DDilatedAsym.h"
#include "Kuiper_KB_Conv2DDilatedAsym.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

// y = conv2d(x, w) + bias  (bias optional; if undefined, zeroed scratch)
static torch::Tensor kuiper_conv2d_dilated_asym_cuda(
        torch::Tensor X, torch::Tensor W,
        c10::optional<torch::Tensor> Bias_opt,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_conv2d_dilated_asym: X and W must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                W.scalar_type() == torch::kFloat32,
                "kuiper_conv2d_dilated_asym: X and W must be float32");
    TORCH_CHECK(X.dim() == 4 && W.dim() == 4,
                "kuiper_conv2d_dilated_asym: X and W must be 4D");
    TORCH_CHECK(groups == 1,
                "kuiper_conv2d_dilated_asym: only groups=1 supported");
    TORCH_CHECK(stride_h >= 1 && stride_w >= 1,
                "kuiper_conv2d_dilated_asym: strides must be >= 1");
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0,
                "kuiper_conv2d_dilated_asym: pads must be >= 0");
    TORCH_CHECK(dil_h >= 1 && dil_w >= 1,
                "kuiper_conv2d_dilated_asym: dilations must be >= 1");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    int64_t B    = Xc.size(0);
    int64_t Cin  = Xc.size(1);
    int64_t Hin  = Xc.size(2);
    int64_t Win  = Xc.size(3);
    int64_t Cout = Wc.size(0);
    int64_t WCin = Wc.size(1);
    int64_t Kh   = Wc.size(2);
    int64_t Kw   = Wc.size(3);
    TORCH_CHECK(Cin == WCin, "kuiper_conv2d_dilated_asym: Cin mismatch");
    int64_t eff_kh = dil_h * (Kh - 1) + 1;
    int64_t eff_kw = dil_w * (Kw - 1) + 1;
    TORCH_CHECK(Hin + 2*pad_h >= eff_kh && Win + 2*pad_w >= eff_kw,
                "kuiper_conv2d_dilated_asym: padded input < dilated kernel");
    int64_t Hout = (int64_t)Kuiper_KB_Conv2DDilatedAsym_conv2dd_out_dim_sz(
        (uint32_t)Hin, (uint32_t)Kh, (uint32_t)stride_h, (uint32_t)dil_h, (uint32_t)pad_h);
    int64_t Wout = (int64_t)Kuiper_KB_Conv2DDilatedAsym_conv2dd_out_dim_sz(
        (uint32_t)Win, (uint32_t)Kw, (uint32_t)stride_w, (uint32_t)dil_w, (uint32_t)pad_w);
    TORCH_CHECK(Hout >= 1 && Wout >= 1,
                "kuiper_conv2d_dilated_asym: zero-sized output");
    int64_t nthr = B * Cout * Hout * Wout;
    int64_t xnel = B * Cin  * Hin  * Win;
    int64_t wnel = Cout * Cin * Kh * Kw;
    int64_t ynel = nthr;
    TORCH_CHECK(B > 0 && Cin > 0 && Hin > 0 && Win > 0 && Cout > 0
                && Kh > 0 && Kw > 0,
                "kuiper_conv2d_dilated_asym: shapes must be positive");
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
                // SZ.fits side conditions (h_out*sh + kh*dh, w_out*sw + kw*dw)
                (Hout*stride_h + Kh*dil_h) <= (int64_t)UINT32_MAX &&
                (Wout*stride_w + Kw*dil_w) <= (int64_t)UINT32_MAX &&
                nthr <= KUIPER_MAX_NTHR,
                "kuiper_conv2d_dilated_asym: shape out of range");

    auto Y = torch::empty({B, Cout, Hout, Wout}, Xc.options());

    float *bias_d = nullptr;
    bool bias_owned = false;
    if (Bias_opt.has_value() && Bias_opt->defined()) {
        auto Bias = Bias_opt->contiguous().to(X.device()).to(torch::kFloat32);
        TORCH_CHECK(Bias.numel() == Cout,
                    "kuiper_conv2d_dilated_asym: bias.numel() != Cout");
        size_t bb = (size_t)Cout * sizeof(float);
        TORCH_CHECK(cudaMalloc(&bias_d, bb) == cudaSuccess,
                    "kuiper_conv2d_dilated_asym: cudaMalloc bias scratch failed");
        TORCH_CHECK(cudaMemcpy(bias_d, Bias.data_ptr<float>(), bb,
                               cudaMemcpyDeviceToDevice) == cudaSuccess,
                    "kuiper_conv2d_dilated_asym: cudaMemcpy bias scratch failed");
        bias_owned = true;
    } else {
        size_t bb = (size_t)Cout * sizeof(float);
        TORCH_CHECK(cudaMalloc(&bias_d, bb) == cudaSuccess,
                    "kuiper_conv2d_dilated_asym: cudaMalloc bias failed");
        TORCH_CHECK(cudaMemset(bias_d, 0, bb) == cudaSuccess,
                    "kuiper_conv2d_dilated_asym: cudaMemset bias failed");
        bias_owned = true;
    }

    Kuiper_KB_Conv2DDilatedAsym_conv2d_dilated_asym_f32(
        (uint32_t)B, (uint32_t)Cin, (uint32_t)Hin, (uint32_t)Win,
        (uint32_t)Cout, (uint32_t)Kh, (uint32_t)Kw,
        (uint32_t)stride_h, (uint32_t)stride_w,
        (uint32_t)pad_h, (uint32_t)pad_w,
        (uint32_t)dil_h, (uint32_t)dil_w,
        (uint32_t)Hout, (uint32_t)Wout,
        Xc.data_ptr<float>(), Wc.data_ptr<float>(),
        bias_d, Y.data_ptr<float>());

    if (bias_owned) cudaFree(bias_d);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_conv2d_dilated_asym", &kuiper_conv2d_dilated_asym_cuda,
          "Kuiper verified Conv2D forward (asymmetric stride/pad/dilation)");
}
