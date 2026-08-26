// Common bridge body for KernelBench L1 #67/#76 — Conv1D forward
// (general): stride, pad, dilation, optional bias.
//
// Each per-challenge bridge `#include`s this file and only differs in
// the PYBIND11_MODULE name plumbing.  The heavy lifting (output-size
// math, output allocation, kernel call) now lives INSIDE the
// verification boundary:
//   * Output dim [Lout] comes from the VERIFIED extracted helper
//     `Kuiper_KB_Conv1DAlloc_conv1d_out_dim` (no C++ division).
//   * The output buffer is allocated INSIDE the verified self-allocating
//     entry `Kuiper_KB_Conv1DAlloc_conv1d_general_alloc_f32` (cudaMalloc
//     via KPR_GPU_ALLOC) and returned as a bare device pointer, which we
//     wrap in a torch::Tensor with a cudaFree deleter (no `torch::empty`).
// This driver therefore performs NO conv output-size arithmetic and NO
// output allocation.  It only checks raw-dimension contracts and handles
// the bias scratch copy/zero (a copy/zero, not algorithmic math).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_Conv1DAlloc.h"
#include "Kuiper_KB_Conv1DAlloc.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

// Forward: y = conv1d(x, w) + bias  (bias optional; if undefined, zeroed scratch is used)
static torch::Tensor kuiper_conv1d_general_cuda(
        torch::Tensor X, torch::Tensor W,
        c10::optional<torch::Tensor> Bias_opt,
        int64_t stride,
        int64_t pad,
        int64_t dilation,
        int64_t groups) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_conv1d_general: X and W must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                W.scalar_type() == torch::kFloat32,
                "kuiper_conv1d_general: X and W must be float32");
    TORCH_CHECK(X.dim() == 3 && W.dim() == 3,
                "kuiper_conv1d_general: X and W must be 3D (N,C,L)");
    TORCH_CHECK(groups == 1,
                "kuiper_conv1d_general: only groups=1 supported");
    TORCH_CHECK(stride   >= 1, "kuiper_conv1d_general: stride must be >= 1");
    TORCH_CHECK(pad      >= 0, "kuiper_conv1d_general: pad must be >= 0");
    TORCH_CHECK(dilation >= 1, "kuiper_conv1d_general: dilation must be >= 1");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    int64_t B    = Xc.size(0);
    int64_t Cin  = Xc.size(1);
    int64_t Lin  = Xc.size(2);
    int64_t Cout = Wc.size(0);
    int64_t WCin = Wc.size(1);
    int64_t Kk   = Wc.size(2);
    TORCH_CHECK(Cin == WCin, "kuiper_conv1d_general: Cin mismatch");
    TORCH_CHECK(B > 0 && Cin > 0 && Lin > 0 && Cout > 0 && Kk > 0,
                "kuiper_conv1d_general: shapes must be positive");

    // Padded input must be at least as large as the dilated (effective)
    // kernel span.  This discharges the `(k-1)*dilation + 1 <= n + 2*pad`
    // precondition of the verified `conv1d_out_dim` (no size_t underflow)
    // AND keeps the padded extent in u32 so the helper does not overflow
    // internally.
    int64_t eff_k = (Kk - 1) * dilation + 1;
    int64_t Lpad  = Lin + 2 * pad;
    TORCH_CHECK(Lpad >= eff_k,
                "kuiper_conv1d_general: padded input smaller than effective kernel");
    TORCH_CHECK(Lpad <= (int64_t)UINT32_MAX,
                "kuiper_conv1d_general: padded input out of u32 range");

    // VERIFIED conv1d output-size division (extracted from Kuiper.KB.Conv1DAlloc):
    //   Lout = (Lin + 2*pad - ((Kk-1)*dilation + 1)) / stride + 1
    // No hand-written C++ division here.
    int64_t Lout = (int64_t)Kuiper_KB_Conv1DAlloc_conv1d_out_dim(
        (uint32_t)Lin, (uint32_t)Kk, (uint32_t)stride, (uint32_t)dilation,
        (uint32_t)pad);
    TORCH_CHECK(Lout >= 1,
                "kuiper_conv1d_general: zero-sized output");

    int64_t nthr = B * Cout * Lout;
    int64_t xnel = B * Cin  * Lin;
    int64_t wnel = Cout * Cin * Kk;
    int64_t ynel = nthr;
    // Raw-dimension contracts discharging the verified `conv1d_size_req`
    // precondition (all multiplications fit u32; nthr within launch bound).
    TORCH_CHECK(B    <= (int64_t)UINT32_MAX &&
                Cin  <= (int64_t)UINT32_MAX &&
                Lin  <= (int64_t)UINT32_MAX &&
                Cout <= (int64_t)UINT32_MAX &&
                Kk   <= (int64_t)UINT32_MAX &&
                Lout <= (int64_t)UINT32_MAX &&
                xnel <= (int64_t)UINT32_MAX &&
                wnel <= (int64_t)UINT32_MAX &&
                ynel <= (int64_t)UINT32_MAX &&
                stride   <= (int64_t)UINT32_MAX &&
                pad      <= (int64_t)UINT32_MAX &&
                dilation <= (int64_t)UINT32_MAX &&
                // SZ.fits side condition (l_out*stride + kk*dilation fits)
                (Lout*stride + Kk*dilation) <= (int64_t)UINT32_MAX &&
                nthr <= KUIPER_MAX_NTHR,
                "kuiper_conv1d_general: shape out of range");

    // Bias: forward caller's tensor if defined, otherwise zeroed scratch.
    // This is a copy / zero-fill, not algorithmic math.
    float *bias_d = nullptr;
    bool bias_owned = false;
    if (Bias_opt.has_value() && Bias_opt->defined()) {
        auto Bias = Bias_opt->contiguous().to(X.device()).to(torch::kFloat32);
        TORCH_CHECK(Bias.numel() == Cout,
                    "kuiper_conv1d_general: bias.numel() != Cout");
        size_t bb = (size_t)Cout * sizeof(float);
        float *scratch = nullptr;
        TORCH_CHECK(cudaMalloc(&scratch, bb) == cudaSuccess,
                    "kuiper_conv1d_general: cudaMalloc bias scratch failed");
        TORCH_CHECK(cudaMemcpy(scratch, Bias.data_ptr<float>(), bb,
                               cudaMemcpyDeviceToDevice) == cudaSuccess,
                    "kuiper_conv1d_general: cudaMemcpy bias scratch failed");
        bias_d = scratch;
        bias_owned = true;
    } else {
        size_t bb = (size_t)Cout * sizeof(float);
        TORCH_CHECK(cudaMalloc(&bias_d, bb) == cudaSuccess,
                    "kuiper_conv1d_general: cudaMalloc bias failed");
        TORCH_CHECK(cudaMemset(bias_d, 0, bb) == cudaSuccess,
                    "kuiper_conv1d_general: cudaMemset bias failed");
        bias_owned = true;
    }

    // Self-allocating verified entry: allocates the (B*Cout*Lout) output
    // buffer inside the verification boundary, runs the verified kernel, and
    // returns the device pointer (ownership passes to us).
    float *out_ptr = Kuiper_KB_Conv1DAlloc_conv1d_general_alloc_f32(
        (uint32_t)B, (uint32_t)Cin, (uint32_t)Lin,
        (uint32_t)Cout, (uint32_t)Kk,
        (uint32_t)stride, (uint32_t)pad, (uint32_t)dilation,
        (uint32_t)Lout,
        Xc.data_ptr<float>(), Wc.data_ptr<float>(), bias_d);

    if (bias_owned) cudaFree(bias_d);

    // Wrap the Kuiper-allocated device buffer in a tensor that owns it: the
    // deleter cudaFree's it when the tensor is destroyed.
    auto Y = torch::from_blob(
        out_ptr, {B, Cout, Lout},
        [](void *p) { cudaFree(p); },
        Xc.options());
    return Y;
}
