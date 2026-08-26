// Bridge for KernelBench L2 #86: Matmul_Divide_GELU.
//
// PyTorch reference (nn.Linear, WITH bias):
//   x = linear(x); x = x / divisor; x = gelu(x)   (exact erf-form GELU)
//   -> (batch, out)
//
// Verified pipeline (single extracted entry, 3 GPU launches):
//   1. GEMM      : gC := x @ wt            (wt = W.T, input x out)
//   2. bias-add  : y[i*out+j] := C[i,j] + bias[j]
//   3. fused map : y := gelu(y / divisor) = 0.5*v*(1+erf(v/sqrt2)), v=y/divisor
// EXACT float-level postcondition (no real approximation).
#include <torch/extension.h>
#include "Kuiper_KB_MatmulDivGelu.h"
#include "Kuiper_KB_MatmulDivGelu.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_mdg_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor B,
                              double divisor) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 2,
                "kuiper_mdg: expected 2-D float32 CUDA tensor X");
    TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kFloat32 && W.dim() == 2,
                "kuiper_mdg: expected 2-D float32 CUDA tensor W");
    TORCH_CHECK(B.is_cuda() && B.scalar_type() == torch::kFloat32 && B.dim() == 1,
                "kuiper_mdg: expected 1-D float32 CUDA tensor bias");
    TORCH_CHECK(X.device() == W.device() && X.device() == B.device(),
                "X, W, bias must be on the same device");

    auto Xc = X.contiguous();                     // (batch, input)
    int64_t batch = Xc.size(0), input = Xc.size(1);
    int64_t out = W.size(0);                       // nn.Linear weight is (out, input)
    TORCH_CHECK(W.size(1) == input,
                "shape mismatch: X is (", batch, ",", input,
                "), W is (", out, ",", W.size(1), ")");
    TORCH_CHECK(B.size(0) == out,
                "shape mismatch: bias is (", B.size(0), "), expected (", out, ")");

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
                "kuiper_mdg: shape out of range for the verified kernel ABI");

    auto Y = torch::empty({batch * out}, Xc.options());
    Kuiper_KB_MatmulDivGelu_matmul_div_gelu_f32(
        (uint32_t)batch, (uint32_t)input, (uint32_t)out,
        (float)divisor,
        Xc.data_ptr<float>(), Wt.data_ptr<float>(),
        Bc.data_ptr<float>(), Y.data_ptr<float>());

    return Y.reshape({batch, out});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_mdg", &kuiper_mdg_cuda,
          "Kuiper verified Matmul/Divide/GELU fusion");
}
