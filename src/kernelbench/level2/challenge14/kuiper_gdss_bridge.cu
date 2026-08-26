// Bridge for KernelBench L2 #14: Gemm_Divide_Sum_Scaling.
//
// PyTorch reference (no bias):
//   x = matmul(x, W.T)            # (batch, hidden)
//   x = x / 2                     # divide
//   x = sum(x, dim=1, keepdim)    # (batch, 1)  -- sum over hidden
//   x = x * scaling_factor        # scale
//
// Verified pipeline (single extracted entry point, 3 GPU launches):
//   1. GEMM:  gC := x @ wt   (wt = W.T, input x hidden)
//   2. Row-sum over hidden into y
//   3. Scalar-multiply y by k = scaling_factor / 2
// giving  y[r] %~ (sum_j (x @ W.T)[r,j]) * (scaling_factor/2)
//        == (sum_j (x @ W.T)[r,j] / 2) * scaling_factor.
#include <torch/extension.h>
#include "Kuiper_KB_GemmDivSumScale.h"
#include "Kuiper_KB_GemmDivSumScale.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_gdss_cuda(torch::Tensor X, torch::Tensor W, double scaling_factor) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 2,
                "kuiper_gdss: expected 2-D float32 CUDA tensor X");
    TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kFloat32 && W.dim() == 2,
                "kuiper_gdss: expected 2-D float32 CUDA tensor W");
    TORCH_CHECK(X.device() == W.device(), "X and W must be on the same device");

    auto Xc = X.contiguous();                     // (batch, input)
    int64_t batch = Xc.size(0), input = Xc.size(1);
    TORCH_CHECK(W.size(1) == input,
                "shape mismatch: X is (", batch, ",", input,
                "), W is (", W.size(0), ",", W.size(1), ")");
    int64_t hidden = W.size(0);

    // Caller-side fact the proof relies on: wt = W.T  (input x hidden).
    auto Wt = W.transpose(0, 1).contiguous();     // (input, hidden)

    TORCH_CHECK(batch > 0 && input > 0 && hidden > 0
                && batch  <= (int64_t)UINT32_MAX
                && input  <= (int64_t)UINT32_MAX
                && hidden <= (int64_t)UINT32_MAX
                && batch * input  <= (int64_t)UINT32_MAX
                && input * hidden <= (int64_t)UINT32_MAX
                && batch * hidden <= (int64_t)UINT32_MAX
                && batch <= KUIPER_MAX_BLOCKS
                && batch * hidden <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS
                && hidden + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX,
                "kuiper_gdss: shape out of range for the verified kernel ABI");

    // k = scaling_factor / 2  (folds divide-by-2 and final scale).
    float k = (float)(scaling_factor / 2.0);

    auto Y = torch::empty({batch}, Xc.options());
    Kuiper_KB_GemmDivSumScale_gemm_div_sum_scale_f32(
        (uint32_t)batch, (uint32_t)input, (uint32_t)hidden,
        k,
        Xc.data_ptr<float>(), Wt.data_ptr<float>(), Y.data_ptr<float>());

    return Y.reshape({batch, 1});                 // keepdim=True
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_gdss", &kuiper_gdss_cuda,
          "Kuiper verified Gemm/Divide/Sum/Scaling fusion");
}
