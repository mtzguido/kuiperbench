// Bridge body for KernelBench L1 #75 — ConvTranspose2D with groups
// dispatched at the host level (loops over groups, calling the verified
// Kuiper.KB.ConvT2DGeneral self-allocating primitive once per group).  The
// verified primitive itself handles asymmetric inputs/kernels, stride, pad,
// dilation, bias, output-size math, AND output allocation for a single
// group.  This driver performs NO ConvTranspose output-size arithmetic:
// output dims come from the VERIFIED extracted helper
// `Kuiper_KB_ConvT2DGeneral_convt_out_dim`.
#include <torch/extension.h>
#include "Kuiper_KB_ConvT2DGeneral.h"
#include "Kuiper_KB_ConvT2DGeneral.cu"

static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

// Forward: y = convT2d(x, w) + bias, with optional groups.
static torch::Tensor kuiper_convt2d_grouped_cuda(
        torch::Tensor X, torch::Tensor W,
        c10::optional<torch::Tensor> Bias_opt,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t out_pad_h, int64_t out_pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_convt2d_grouped: X and W must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                W.scalar_type() == torch::kFloat32,
                "kuiper_convt2d_grouped: X and W must be float32");
    TORCH_CHECK(X.dim() == 4 && W.dim() == 4,
                "kuiper_convt2d_grouped: X and W must be 4D");
    TORCH_CHECK(groups >= 1, "kuiper_convt2d_grouped: groups must be >= 1");
    TORCH_CHECK(stride_h >= 1 && stride_w >= 1,
                "kuiper_convt2d_grouped: stride must be >= 1");
    TORCH_CHECK(pad_h >= 0 && pad_w >= 0,
                "kuiper_convt2d_grouped: pad must be >= 0");
    TORCH_CHECK(out_pad_h >= 0 && out_pad_w >= 0,
                "kuiper_convt2d_grouped: output_padding must be >= 0");
    TORCH_CHECK(dil_h >= 1 && dil_w >= 1,
                "kuiper_convt2d_grouped: dilation must be >= 1");
    TORCH_CHECK((out_pad_h < stride_h || out_pad_h < dil_h) &&
                (out_pad_w < stride_w || out_pad_w < dil_w),
                "kuiper_convt2d_grouped: output_padding must be < max(stride, dilation)");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    int64_t B    = Xc.size(0);
    int64_t Cin  = Xc.size(1);
    int64_t Hin  = Xc.size(2);
    int64_t Win  = Xc.size(3);
    int64_t WCin = Wc.size(0);          // ConvT layout: (Cin, Cout/G, kH, kW)
    int64_t Cout_pg = Wc.size(1);
    int64_t Kh   = Wc.size(2);
    int64_t Kw   = Wc.size(3);
    TORCH_CHECK(Cin == WCin,
                "kuiper_convt2d_grouped: weight Cin axis mismatch");
    TORCH_CHECK(Cin % groups == 0,
                "kuiper_convt2d_grouped: Cin must be divisible by groups");
    int64_t Cin_pg = Cin / groups;
    int64_t Cout = Cout_pg * groups;
    TORCH_CHECK(B > 0 && Cin > 0 && Hin > 0 && Win > 0 && Cout > 0
                && Kh > 0 && Kw > 0,
                "kuiper_convt2d_grouped: shapes must be positive");

    // VERIFIED ConvTranspose2D output-size formula (extracted from
    // Kuiper.KB.ConvT2DGeneral), per axis.  [pos] is the non-negative part
    // used only as an underflow/overflow guard (NOT the output value); the
    // actual output dimensions come from the verified helper.
    int64_t pos_h = (Hin - 1) * stride_h + dil_h * (Kh - 1) + out_pad_h + 1;
    int64_t pos_w = (Win - 1) * stride_w + dil_w * (Kw - 1) + out_pad_w + 1;
    TORCH_CHECK(pos_h > 2 * pad_h && pos_w > 2 * pad_w,
                "kuiper_convt2d_grouped: zero-sized output");
    TORCH_CHECK(pos_h <= (int64_t)UINT32_MAX && pos_w <= (int64_t)UINT32_MAX,
                "kuiper_convt2d_grouped: output-size intermediate out of u32 range");
    int64_t Hout = (int64_t)Kuiper_KB_ConvT2DGeneral_convt_out_dim(
        (uint32_t)Hin, (uint32_t)stride_h, (uint32_t)dil_h, (uint32_t)Kh,
        (uint32_t)pad_h, (uint32_t)out_pad_h);
    int64_t Wout = (int64_t)Kuiper_KB_ConvT2DGeneral_convt_out_dim(
        (uint32_t)Win, (uint32_t)stride_w, (uint32_t)dil_w, (uint32_t)Kw,
        (uint32_t)pad_w, (uint32_t)out_pad_w);

    int64_t nthr_pg = B * Cout_pg * Hout * Wout;
    auto fits_u32 = [](int64_t v) { return v >= 0 && v <= (int64_t)UINT32_MAX; };
    TORCH_CHECK(fits_u32(B) && fits_u32(Cin) && fits_u32(Cin_pg) &&
                fits_u32(Hin) && fits_u32(Win) &&
                fits_u32(Cout) && fits_u32(Cout_pg) &&
                fits_u32(Kh) && fits_u32(Kw) &&
                fits_u32(Hout) && fits_u32(Wout) &&
                fits_u32(stride_h) && fits_u32(stride_w) &&
                fits_u32(pad_h) && fits_u32(pad_w) &&
                fits_u32(dil_h) && fits_u32(dil_w) &&
                fits_u32(B * Cin_pg * Hin * Win) &&
                fits_u32(Cin_pg * Cout_pg * Kh * Kw) &&
                fits_u32(nthr_pg) &&
                fits_u32(Kh * Kw) && fits_u32(Cin_pg * Kh * Kw) &&
                fits_u32(Hout * Wout) && fits_u32(Cout_pg * Hout * Wout) &&
                fits_u32(Hout + pad_h) && fits_u32(Wout + pad_w) &&
                fits_u32(Kh * dil_h) && fits_u32(Kw * dil_w) &&
                nthr_pg <= KUIPER_MAX_NTHR,
                "kuiper_convt2d_grouped: shape out of range");

    float *bias_full_d = nullptr;
    bool bias_owned = false;
    size_t bb = (size_t)Cout * sizeof(float);
    if (Bias_opt.has_value() && Bias_opt->defined()) {
        auto Bias = Bias_opt->contiguous().to(X.device()).to(torch::kFloat32);
        TORCH_CHECK(Bias.numel() == Cout,
                    "kuiper_convt2d_grouped: bias.numel() != Cout");
        TORCH_CHECK(cudaMalloc(&bias_full_d, bb) == cudaSuccess,
                    "kuiper_convt2d_grouped: cudaMalloc bias scratch failed");
        TORCH_CHECK(cudaMemcpy(bias_full_d, Bias.data_ptr<float>(), bb,
                               cudaMemcpyDeviceToDevice) == cudaSuccess,
                    "kuiper_convt2d_grouped: cudaMemcpy bias scratch failed");
        bias_owned = true;
    } else {
        TORCH_CHECK(cudaMalloc(&bias_full_d, bb) == cudaSuccess,
                    "kuiper_convt2d_grouped: cudaMalloc bias failed");
        TORCH_CHECK(cudaMemset(bias_full_d, 0, bb) == cudaSuccess,
                    "kuiper_convt2d_grouped: cudaMemset bias failed");
        bias_owned = true;
    }

    // groups == 1 fast path: one self-allocating verified call + from_blob.
    if (groups == 1) {
        float *out_ptr = Kuiper_KB_ConvT2DGeneral_convt2d_general_alloc_f32(
            (uint32_t)B, (uint32_t)Cin_pg, (uint32_t)Hin, (uint32_t)Win,
            (uint32_t)Cout_pg, (uint32_t)Kh, (uint32_t)Kw,
            (uint32_t)stride_h, (uint32_t)stride_w,
            (uint32_t)pad_h, (uint32_t)pad_w,
            (uint32_t)dil_h, (uint32_t)dil_w,
            (uint32_t)Hout, (uint32_t)Wout,
            Xc.data_ptr<float>(), Wc.data_ptr<float>(), bias_full_d);
        if (bias_owned) cudaFree(bias_full_d);
        return torch::from_blob(
            out_ptr, {B, Cout, Hout, Wout},
            [](void *p) { cudaFree(p); }, Xc.options());
    }

    // Grouped path: assemble per-group verified slabs into the full output.
    auto Y = torch::empty({B, Cout, Hout, Wout}, Xc.options());
    for (int64_t g = 0; g < groups; ++g) {
        auto X_g = Xc.slice(/*dim=*/1, g * Cin_pg, (g + 1) * Cin_pg).contiguous();
        auto W_g = Wc.slice(/*dim=*/0, g * Cin_pg, (g + 1) * Cin_pg).contiguous();
        float *bias_g = bias_full_d + g * Cout_pg;

        float *out_ptr_g = Kuiper_KB_ConvT2DGeneral_convt2d_general_alloc_f32(
            (uint32_t)B, (uint32_t)Cin_pg, (uint32_t)Hin, (uint32_t)Win,
            (uint32_t)Cout_pg, (uint32_t)Kh, (uint32_t)Kw,
            (uint32_t)stride_h, (uint32_t)stride_w,
            (uint32_t)pad_h, (uint32_t)pad_w,
            (uint32_t)dil_h, (uint32_t)dil_w,
            (uint32_t)Hout, (uint32_t)Wout,
            X_g.data_ptr<float>(), W_g.data_ptr<float>(), bias_g);

        auto Y_g = torch::from_blob(
            out_ptr_g, {B, Cout_pg, Hout, Wout},
            [](void *p) { cudaFree(p); }, Xc.options());
        Y.slice(/*dim=*/1, g * Cout_pg, (g + 1) * Cout_pg).copy_(Y_g);
    }

    if (bias_owned) cudaFree(bias_full_d);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_convt2d_grouped", &kuiper_convt2d_grouped_cuda,
          "Kuiper verified ConvTranspose2D forward with host-side groups dispatch");
}
