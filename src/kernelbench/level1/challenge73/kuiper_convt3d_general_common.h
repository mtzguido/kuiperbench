// Common bridge body for KernelBench L1 #58/#61/#68/#70/#72/#73/#77 —
// ConvTranspose3D forward (general): asymmetric inputs/kernels, stride,
// pad, output_padding, dilation, groups, optional bias.  Each per-
// challenge bridge `#define`s a unique TORCH_EXTENSION_NAME and
// `#include`s this header.
//
// The output-size math and output allocation now live INSIDE the
// verification boundary:
//   * Output dims [Dout]/[Hout]/[Wout] come from the VERIFIED extracted
//     helper `Kuiper_KB_ConvT3DGeneral_convt_out_dim` (no C++ formula).
//   * The (per-group) output buffer is allocated INSIDE the verified
//     self-allocating entry
//     `Kuiper_KB_ConvT3DGeneral_convt3d_general_alloc_f32` and returned as
//     a bare device pointer, wrapped in a torch::Tensor with a cudaFree
//     deleter (no `torch::empty` for the verified slab).
// This driver performs NO ConvTranspose output-size arithmetic.  Grouped
// variants are dispatched at the host level: slice channels per group, run
// the verified self-allocating primitive once per group, and write results
// back.  groups=1 falls through to a single primitive call with from_blob.
#include <torch/extension.h>
#include "Kuiper_KB_ConvT3DGeneral.h"
#include "Kuiper_KB_ConvT3DGeneral.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

// Forward: y = convT3d(x, w) + bias
static torch::Tensor kuiper_convt3d_general_cuda(
        torch::Tensor X, torch::Tensor W,
        c10::optional<torch::Tensor> Bias_opt,
        int64_t stride_d, int64_t stride_h, int64_t stride_w,
        int64_t pad_d, int64_t pad_h, int64_t pad_w,
        int64_t out_pad_d, int64_t out_pad_h, int64_t out_pad_w,
        int64_t dil_d, int64_t dil_h, int64_t dil_w,
        int64_t groups) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_convt3d_general: X and W must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                W.scalar_type() == torch::kFloat32,
                "kuiper_convt3d_general: X and W must be float32");
    TORCH_CHECK(X.dim() == 5 && W.dim() == 5,
                "kuiper_convt3d_general: X and W must be 5D");
    TORCH_CHECK(groups >= 1, "kuiper_convt3d_general: groups must be >= 1");
    TORCH_CHECK(stride_d >= 1 && stride_h >= 1 && stride_w >= 1,
                "kuiper_convt3d_general: stride must be >= 1");
    TORCH_CHECK(pad_d >= 0 && pad_h >= 0 && pad_w >= 0,
                "kuiper_convt3d_general: pad must be >= 0");
    TORCH_CHECK(out_pad_d >= 0 && out_pad_h >= 0 && out_pad_w >= 0,
                "kuiper_convt3d_general: output_padding must be >= 0");
    TORCH_CHECK(dil_d >= 1 && dil_h >= 1 && dil_w >= 1,
                "kuiper_convt3d_general: dilation must be >= 1");
    // PyTorch's ConvTranspose invariant: each output_padding axis must be
    // strictly less than the corresponding stride OR dilation.
    TORCH_CHECK((out_pad_d < stride_d || out_pad_d < dil_d) &&
                (out_pad_h < stride_h || out_pad_h < dil_h) &&
                (out_pad_w < stride_w || out_pad_w < dil_w),
                "kuiper_convt3d_general: output_padding must be < max(stride, dilation)");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    int64_t B    = Xc.size(0);
    int64_t Cin  = Xc.size(1);
    int64_t Din  = Xc.size(2);
    int64_t Hin  = Xc.size(3);
    int64_t Win  = Xc.size(4);
    int64_t WCin = Wc.size(0);          // ConvT layout: (Cin, Cout/G, kD, kH, kW)
    int64_t Cout_pg = Wc.size(1);       // out channels per group
    int64_t Kd   = Wc.size(2);
    int64_t Kh   = Wc.size(3);
    int64_t Kw   = Wc.size(4);
    TORCH_CHECK(Cin == WCin,
                "kuiper_convt3d_general: weight Cin axis mismatch (X.size(1) != W.size(0))");
    TORCH_CHECK(Cin % groups == 0,
                "kuiper_convt3d_general: Cin must be divisible by groups");
    int64_t Cin_pg = Cin / groups;
    int64_t Cout = Cout_pg * groups;
    TORCH_CHECK(B > 0 && Cin > 0 && Din > 0 && Hin > 0 && Win > 0 &&
                Cout > 0 && Kd > 0 && Kh > 0 && Kw > 0,
                "kuiper_convt3d_general: shapes must be positive");
    auto helper_arg_fits = [](int64_t v) {
        return v >= 0 && v <= (int64_t)UINT32_MAX;
    };
    TORCH_CHECK(helper_arg_fits(Din) && helper_arg_fits(Hin) && helper_arg_fits(Win) &&
                helper_arg_fits(Kd) && helper_arg_fits(Kh) && helper_arg_fits(Kw) &&
                helper_arg_fits(stride_d) && helper_arg_fits(stride_h) && helper_arg_fits(stride_w) &&
                helper_arg_fits(pad_d) && helper_arg_fits(pad_h) && helper_arg_fits(pad_w) &&
                helper_arg_fits(out_pad_d) && helper_arg_fits(out_pad_h) && helper_arg_fits(out_pad_w) &&
                helper_arg_fits(dil_d) && helper_arg_fits(dil_h) && helper_arg_fits(dil_w),
                "kuiper_convt3d_general: output-size helper arguments out of range");

    // VERIFIED ConvTranspose3D output-size formula (extracted from
    // Kuiper.KB.ConvT3DGeneral), per axis:
    //   L_out = (L_in - 1)*S - 2*P + D*(K - 1) + output_padding + 1.
    // The non-negative part [pos] is computed in a wide integer as an
    // underflow/overflow guard (NOT the output value): [pos > 2*P]
    // discharges the verified helper's [2*P <= pos] precondition and
    // [pos <= UINT32_MAX] discharges its [SZ.fits] precondition.  The actual
    // output dimensions are produced by the verified helper.
    __int128 pos_d = (__int128)(Din - 1) * stride_d +
                     (__int128)dil_d * (Kd - 1) + out_pad_d + 1;
    __int128 pos_h = (__int128)(Hin - 1) * stride_h +
                     (__int128)dil_h * (Kh - 1) + out_pad_h + 1;
    __int128 pos_w = (__int128)(Win - 1) * stride_w +
                     (__int128)dil_w * (Kw - 1) + out_pad_w + 1;
    TORCH_CHECK(pos_d > 2 * pad_d && pos_h > 2 * pad_h && pos_w > 2 * pad_w,
                "kuiper_convt3d_general: zero-sized output");
    TORCH_CHECK(pos_d <= (int64_t)UINT32_MAX && pos_h <= (int64_t)UINT32_MAX &&
                pos_w <= (int64_t)UINT32_MAX,
                "kuiper_convt3d_general: output-size intermediate out of u32 range");
    int64_t Dout = (int64_t)Kuiper_KB_ConvT3DGeneral_convt_out_dim(
        (uint32_t)Din, (uint32_t)stride_d, (uint32_t)dil_d, (uint32_t)Kd,
        (uint32_t)pad_d, (uint32_t)out_pad_d);
    int64_t Hout = (int64_t)Kuiper_KB_ConvT3DGeneral_convt_out_dim(
        (uint32_t)Hin, (uint32_t)stride_h, (uint32_t)dil_h, (uint32_t)Kh,
        (uint32_t)pad_h, (uint32_t)out_pad_h);
    int64_t Wout = (int64_t)Kuiper_KB_ConvT3DGeneral_convt_out_dim(
        (uint32_t)Win, (uint32_t)stride_w, (uint32_t)dil_w, (uint32_t)Kw,
        (uint32_t)pad_w, (uint32_t)out_pad_w);

    int64_t nthr_pg = B * Cout_pg * Dout * Hout * Wout;
    int64_t xnel_pg = B * Cin_pg  * Din  * Hin  * Win;
    int64_t wnel_pg = Cin_pg * Cout_pg * Kd * Kh * Kw;

    auto fits_u32 = [](int64_t v) { return v >= 0 && v <= (int64_t)UINT32_MAX; };
    TORCH_CHECK(fits_u32(B) && fits_u32(Cin) && fits_u32(Din) && fits_u32(Hin) && fits_u32(Win) &&
                fits_u32(Cout) && fits_u32(Cout_pg) && fits_u32(Cin_pg) &&
                fits_u32(Kd) && fits_u32(Kh) && fits_u32(Kw) &&
                fits_u32(Dout) && fits_u32(Hout) && fits_u32(Wout) &&
                fits_u32(stride_d) && fits_u32(stride_h) && fits_u32(stride_w) &&
                fits_u32(pad_d) && fits_u32(pad_h) && fits_u32(pad_w) &&
                fits_u32(dil_d) && fits_u32(dil_h) && fits_u32(dil_w) &&
                fits_u32(xnel_pg) && fits_u32(wnel_pg) && fits_u32(nthr_pg) &&
                // SZ.fits side conditions (per-group)
                fits_u32(Kh * Kw) && fits_u32(Kd * Kh * Kw) &&
                fits_u32(Cin_pg * Kd * Kh * Kw) &&
                fits_u32(Hout * Wout) && fits_u32(Dout * Hout * Wout) &&
                fits_u32(Cout_pg * Dout * Hout * Wout) &&
                fits_u32(Dout + pad_d) && fits_u32(Hout + pad_h) && fits_u32(Wout + pad_w) &&
                fits_u32(Kd * dil_d) && fits_u32(Kh * dil_h) && fits_u32(Kw * dil_w) &&
                nthr_pg <= KUIPER_MAX_NTHR,
                "kuiper_convt3d_general: shape out of range");

    // Bias scratch: per-group slice (Cout_pg consecutive elements).
    float *bias_full_d = nullptr;
    bool bias_owned = false;
    size_t bb = (size_t)Cout * sizeof(float);
    if (Bias_opt.has_value() && Bias_opt->defined()) {
        auto Bias = Bias_opt->contiguous().to(X.device()).to(torch::kFloat32);
        TORCH_CHECK(Bias.numel() == Cout,
                    "kuiper_convt3d_general: bias.numel() != Cout");
        TORCH_CHECK(cudaMalloc(&bias_full_d, bb) == cudaSuccess,
                    "kuiper_convt3d_general: cudaMalloc bias scratch failed");
        TORCH_CHECK(cudaMemcpy(bias_full_d, Bias.data_ptr<float>(), bb,
                               cudaMemcpyDeviceToDevice) == cudaSuccess,
                    "kuiper_convt3d_general: cudaMemcpy bias scratch failed");
        bias_owned = true;
    } else {
        TORCH_CHECK(cudaMalloc(&bias_full_d, bb) == cudaSuccess,
                    "kuiper_convt3d_general: cudaMalloc bias failed");
        TORCH_CHECK(cudaMemset(bias_full_d, 0, bb) == cudaSuccess,
                    "kuiper_convt3d_general: cudaMemset bias failed");
        bias_owned = true;
    }

    // groups == 1 fast path: one self-allocating verified call + from_blob,
    // no extra copy.
    if (groups == 1) {
        float *out_ptr = Kuiper_KB_ConvT3DGeneral_convt3d_general_alloc_f32(
            (uint32_t)B, (uint32_t)Cin,
            (uint32_t)Din, (uint32_t)Hin, (uint32_t)Win,
            (uint32_t)Cout_pg,
            (uint32_t)Kd, (uint32_t)Kh, (uint32_t)Kw,
            (uint32_t)stride_d, (uint32_t)stride_h, (uint32_t)stride_w,
            (uint32_t)pad_d, (uint32_t)pad_h, (uint32_t)pad_w,
            (uint32_t)dil_d, (uint32_t)dil_h, (uint32_t)dil_w,
            (uint32_t)Dout, (uint32_t)Hout, (uint32_t)Wout,
            Xc.data_ptr<float>(), Wc.data_ptr<float>(), bias_full_d);
        if (bias_owned) cudaFree(bias_full_d);
        return torch::from_blob(
            out_ptr, {B, Cout, Dout, Hout, Wout},
            [](void *p) { cudaFree(p); }, Xc.options());
    }

    // Grouped path: assemble per-group verified slabs into the full output.
    auto Y = torch::empty({B, Cout, Dout, Hout, Wout}, Xc.options());
    for (int64_t g = 0; g < groups; ++g) {
        // Slice X / W along Cin: groups consecutive Cin_pg channels.
        auto X_g = Xc.slice(/*dim=*/1, g * Cin_pg, (g + 1) * Cin_pg).contiguous();
        auto W_g = Wc.slice(/*dim=*/0, g * Cin_pg, (g + 1) * Cin_pg).contiguous();
        float *bias_g = bias_full_d + g * Cout_pg;

        // Self-allocating verified entry for this group's slab.
        float *out_ptr_g = Kuiper_KB_ConvT3DGeneral_convt3d_general_alloc_f32(
            (uint32_t)B, (uint32_t)Cin_pg,
            (uint32_t)Din, (uint32_t)Hin, (uint32_t)Win,
            (uint32_t)Cout_pg,
            (uint32_t)Kd, (uint32_t)Kh, (uint32_t)Kw,
            (uint32_t)stride_d, (uint32_t)stride_h, (uint32_t)stride_w,
            (uint32_t)pad_d, (uint32_t)pad_h, (uint32_t)pad_w,
            (uint32_t)dil_d, (uint32_t)dil_h, (uint32_t)dil_w,
            (uint32_t)Dout, (uint32_t)Hout, (uint32_t)Wout,
            X_g.data_ptr<float>(), W_g.data_ptr<float>(), bias_g);

        auto Y_g = torch::from_blob(
            out_ptr_g, {B, Cout_pg, Dout, Hout, Wout},
            [](void *p) { cudaFree(p); }, Xc.options());
        // Write back to the full Y buffer.
        Y.slice(/*dim=*/1, g * Cout_pg, (g + 1) * Cout_pg).copy_(Y_g);
    }

    if (bias_owned) cudaFree(bias_full_d);
    return Y;
}
