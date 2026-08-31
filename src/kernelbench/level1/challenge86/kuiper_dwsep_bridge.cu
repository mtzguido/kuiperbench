// L1 #86 bridge: depthwise-separable 2D conv = pointwise(depthwise(x)).
//
// This routes through the *composed* verified entry
// `Kuiper_KB_SeparableConv2D_separable_alloc_f32`, which:
//   - self-allocates the depthwise-output scratch buffer and the final
//     output buffer inside the verification boundary,
//   - runs the verified depthwise kernel then the verified 1x1 (pointwise)
//     general-conv kernel,
//   - proves each output cell equals `separable_out_at` — the per-pixel
//     evaluation of the WHOLE `Kuiper.Spec.SeparableConv2D.separable_conv2d`
//     spec applied to the original inputs (not merely two independent posts).
//
// The output spatial size is computed by the VERIFIED helper
// `Kuiper_KB_SeparableConv2D_separable_out_dim` (no hand-rolled C++
// division).  The bridge only does dim checks, bias scratch copy/zero
// (a memcpy/memset, not algorithmic math), a single verified call, and a
// `torch::from_blob` wrap with a cudaFree deleter (no `torch::empty`).
#include <torch/extension.h>
#include "Kuiper_KB_SeparableConv2D.h"
#include "Kuiper_KB_SeparableConv2D.cu"

static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

static torch::Tensor kuiper_dwsep_cuda(
        torch::Tensor X, torch::Tensor Wdw, torch::Tensor Wpw,
        c10::optional<torch::Tensor> BiasDw_opt,
        c10::optional<torch::Tensor> BiasPw_opt,
        int64_t stride, int64_t pad, int64_t dilation) {
    TORCH_CHECK(X.is_cuda() && Wdw.is_cuda() && Wpw.is_cuda(),
                "kuiper_dwsep: all tensors must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                Wdw.scalar_type() == torch::kFloat32 &&
                Wpw.scalar_type() == torch::kFloat32,
                "kuiper_dwsep: all tensors must be float32");
    TORCH_CHECK(X.dim() == 4 && Wdw.dim() == 4 && Wpw.dim() == 4,
                "kuiper_dwsep: tensors must be 4D");
    TORCH_CHECK(dilation == 1, "kuiper_dwsep: only dilation=1 supported");
    TORCH_CHECK(stride >= 1, "kuiper_dwsep: stride >= 1");
    TORCH_CHECK(pad >= 0,    "kuiper_dwsep: pad >= 0");

    auto Xc = X.contiguous();
    auto Wdwc = Wdw.contiguous();
    auto Wpwc = Wpw.contiguous();
    int64_t B    = Xc.size(0);
    int64_t Cin  = Xc.size(1);
    int64_t Hin  = Xc.size(2);
    int64_t Win  = Xc.size(3);
    TORCH_CHECK(Wdwc.size(0) == Cin && Wdwc.size(1) == 1,
                "kuiper_dwsep: depthwise weight shape (C, 1, kH, kW) required");
    int64_t Kh = Wdwc.size(2);
    int64_t Kw = Wdwc.size(3);
    int64_t Cout  = Wpwc.size(0);
    int64_t PwCin = Wpwc.size(1);
    int64_t PwKh  = Wpwc.size(2);
    int64_t PwKw  = Wpwc.size(3);
    TORCH_CHECK(PwCin == Cin && PwKh == 1 && PwKw == 1,
                "kuiper_dwsep: pointwise weight shape (Cout, Cin, 1, 1) required");
    TORCH_CHECK(B > 0 && Cin > 0 && Cout > 0 && Hin > 0 && Win > 0 &&
                Kh > 0 && Kw > 0,
                "kuiper_dwsep: shapes must be positive");
    TORCH_CHECK(B <= (int64_t)UINT32_MAX && Cin <= (int64_t)UINT32_MAX &&
                Cout <= (int64_t)UINT32_MAX &&
                Hin <= (int64_t)UINT32_MAX && Win <= (int64_t)UINT32_MAX &&
                Kh <= (int64_t)UINT32_MAX && Kw <= (int64_t)UINT32_MAX &&
                stride <= (int64_t)UINT32_MAX && pad <= (int64_t)UINT32_MAX &&
                Hin + 2*pad <= (int64_t)UINT32_MAX &&
                Win + 2*pad <= (int64_t)UINT32_MAX,
                "kuiper_dwsep: output-size helper precondition failed");

    // Precondition of the verified out-dim helper: padded input >= kernel.
    TORCH_CHECK(Hin + 2*pad >= Kh && Win + 2*pad >= Kw,
                "kuiper_dwsep: padded input < kernel");

    // VERIFIED output spatial dimensions (no C++ division).  The depthwise
    // stage fixes the output size; the pointwise (1x1) stage preserves it.
    int64_t Hout = (int64_t)Kuiper_KB_SeparableConv2D_separable_out_dim(
        (uint32_t)Hin, (uint32_t)Kh, (uint32_t)stride, (uint32_t)pad);
    int64_t Wout = (int64_t)Kuiper_KB_SeparableConv2D_separable_out_dim(
        (uint32_t)Win, (uint32_t)Kw, (uint32_t)stride, (uint32_t)pad);
    TORCH_CHECK(Hout >= 1 && Wout >= 1, "kuiper_dwsep: zero-sized intermediate");

    int64_t nthr_dw = B * Cin * Hout * Wout;
    int64_t nthr_pw = B * Cout * Hout * Wout;
    TORCH_CHECK(B    <= (int64_t)UINT32_MAX &&
                Cin  <= (int64_t)UINT32_MAX &&
                Cout <= (int64_t)UINT32_MAX &&
                Hin  <= (int64_t)UINT32_MAX && Win  <= (int64_t)UINT32_MAX &&
                Hout <= (int64_t)UINT32_MAX && Wout <= (int64_t)UINT32_MAX &&
                Kh   <= (int64_t)UINT32_MAX && Kw   <= (int64_t)UINT32_MAX &&
                stride <= (int64_t)UINT32_MAX && pad <= (int64_t)UINT32_MAX &&
                (Hout*stride + Kh) <= (int64_t)UINT32_MAX &&
                (Wout*stride + Kw) <= (int64_t)UINT32_MAX &&
                (B*Cin*Hin*Win)    <= (int64_t)UINT32_MAX &&
                (B*Cin*Hout*Wout)  <= (int64_t)UINT32_MAX &&
                (B*Cout*Hout*Wout) <= (int64_t)UINT32_MAX &&
                nthr_dw <= KUIPER_MAX_NTHR && nthr_pw <= KUIPER_MAX_NTHR,
                "kuiper_dwsep: shape out of range");

    // ----- depthwise bias scratch (copy if provided, else zeroed) -----
    float *bias_dw = nullptr;
    {
        size_t bb = (size_t)Cin * sizeof(float);
        TORCH_CHECK(cudaMalloc(&bias_dw, bb) == cudaSuccess,
                    "kuiper_dwsep: cudaMalloc bias_dw failed");
        if (BiasDw_opt.has_value() && BiasDw_opt->defined()) {
            auto Bd = BiasDw_opt->contiguous().to(X.device()).to(torch::kFloat32);
            TORCH_CHECK(Bd.numel() == Cin, "kuiper_dwsep: dw bias size");
            TORCH_CHECK(cudaMemcpy(bias_dw, Bd.data_ptr<float>(), bb,
                                   cudaMemcpyDeviceToDevice) == cudaSuccess,
                        "kuiper_dwsep: cudaMemcpy bias_dw failed");
        } else {
            TORCH_CHECK(cudaMemset(bias_dw, 0, bb) == cudaSuccess,
                        "kuiper_dwsep: cudaMemset bias_dw failed");
        }
    }

    // ----- pointwise bias scratch (copy if provided, else zeroed) -----
    float *bias_pw = nullptr;
    {
        size_t bb = (size_t)Cout * sizeof(float);
        TORCH_CHECK(cudaMalloc(&bias_pw, bb) == cudaSuccess,
                    "kuiper_dwsep: cudaMalloc bias_pw failed");
        if (BiasPw_opt.has_value() && BiasPw_opt->defined()) {
            auto Bp = BiasPw_opt->contiguous().to(X.device()).to(torch::kFloat32);
            TORCH_CHECK(Bp.numel() == Cout, "kuiper_dwsep: pw bias size");
            TORCH_CHECK(cudaMemcpy(bias_pw, Bp.data_ptr<float>(), bb,
                                   cudaMemcpyDeviceToDevice) == cudaSuccess,
                        "kuiper_dwsep: cudaMemcpy bias_pw failed");
        } else {
            TORCH_CHECK(cudaMemset(bias_pw, 0, bb) == cudaSuccess,
                        "kuiper_dwsep: cudaMemset bias_pw failed");
        }
    }

    // Single composed verified call: allocates scratch + output internally,
    // runs depthwise then pointwise, returns the device output pointer whose
    // contents equal the WHOLE separable-conv spec.
    float *out_ptr = Kuiper_KB_SeparableConv2D_separable_alloc_f32(
        (uint32_t)B, (uint32_t)Cin, (uint32_t)Hin, (uint32_t)Win,
        (uint32_t)Kh, (uint32_t)Kw,
        (uint32_t)stride, (uint32_t)pad,
        (uint32_t)Cout, (uint32_t)Hout, (uint32_t)Wout,
        Xc.data_ptr<float>(), Wdwc.data_ptr<float>(), bias_dw,
        Wpwc.data_ptr<float>(), bias_pw);

    cudaFree(bias_dw);
    cudaFree(bias_pw);

    // Wrap the Kuiper-allocated device buffer in a tensor that owns it.
    auto Y = torch::from_blob(
        out_ptr, {B, Cout, Hout, Wout},
        [](void *p) { cudaFree(p); },
        Xc.options());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_dwsep", &kuiper_dwsep_cuda,
          "Kuiper verified depthwise-separable Conv2D forward (composed spec)");
}
