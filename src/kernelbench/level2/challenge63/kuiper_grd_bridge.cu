// Bridge for KernelBench L2 #63: Gemm_ReLU_Divide.
//
// PyTorch reference (nn.Linear, WITH bias):
//   x = linear(x)        # x @ W.T + bias   -> (batch, out)
//   x = relu(x)          # max(x, 0)
//   x = x / divisor      # divide by a constant
// Output shape (batch, out).
//
// Verified pipeline (single extracted entry point, 4 GPU launches):
//   1. GEMM      : gC := x @ wt            (wt = W.T, input x out)
//   2. bias-add  : y[i*out+j] := C[i,j] + bias[j]
//   3. ReLU      : y := max(y, 0)
//   4. divide    : y := y / divisor
// giving  y[i*out+j] == div (relu ((x @ W.T)[i,j] + bias[j])) divisor
// (EXACT float-level postcondition, no real-number approximation).
#include <torch/extension.h>
#include "Kuiper_KB_GemmReluDivide.h"
#include "Kuiper_KB_GemmReluDivide.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_grd_cuda(torch::Tensor X, torch::Tensor W,
                              torch::Tensor B, double divisor) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 2,
                "kuiper_grd: expected 2-D float32 CUDA tensor X");
    TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kFloat32 && W.dim() == 2,
                "kuiper_grd: expected 2-D float32 CUDA tensor W");
    TORCH_CHECK(B.is_cuda() && B.scalar_type() == torch::kFloat32 && B.dim() == 1,
                "kuiper_grd: expected 1-D float32 CUDA tensor bias");
    TORCH_CHECK(X.device() == W.device() && X.device() == B.device(),
                "X, W, bias must be on the same device");

    auto Xc = X.contiguous();                     // (batch, input)
    int64_t batch = Xc.size(0), input = Xc.size(1);
    // nn.Linear weight is (out, input); reference computes x @ W.T + bias.
    int64_t out = W.size(0);
    TORCH_CHECK(W.size(1) == input,
                "shape mismatch: X is (", batch, ",", input,
                "), W is (", out, ",", W.size(1), ")");
    TORCH_CHECK(B.size(0) == out,
                "shape mismatch: bias is (", B.size(0), "), expected (", out, ")");

    // Caller-side fact the proof relies on: wt = W.T  (input x out).
    auto Wt = W.transpose(0, 1).contiguous();     // (input, out)
    auto Bc = B.contiguous();

    TORCH_CHECK(batch > 0 && input > 0 && out > 0
                && batch <= (int64_t)UINT32_MAX
                && input <= (int64_t)UINT32_MAX
                && out   <= (int64_t)UINT32_MAX
                && batch * input <= (int64_t)UINT32_MAX
                && input * out   <= (int64_t)UINT32_MAX
                && batch * out   <= (int64_t)UINT32_MAX
                && batch * out   <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS,
                "kuiper_grd: shape out of range for the verified kernel ABI");

    auto Y = torch::empty({batch * out}, Xc.options());
    Kuiper_KB_GemmReluDivide_gemm_relu_divide_f32(
        (uint32_t)batch, (uint32_t)input, (uint32_t)out,
        (float)divisor,
        Xc.data_ptr<float>(), Wt.data_ptr<float>(),
        Bc.data_ptr<float>(), Y.data_ptr<float>());

    return Y.reshape({batch, out});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_grd", &kuiper_grd_cuda,
          "Kuiper verified Gemm/ReLU/Divide fusion");
}
