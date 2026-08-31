// Bridge for KernelBench L1 #63: square 2D convolution forward (bias=False).
//
// PyTorch reference: nn.Conv2d(C_in, C_out, K, stride=1, padding=0, dilation=1, bias=False)
// Input  X: (B, C_in,  H_in,  W_in=H_in)
// Weight W: (C_out, C_in, K, K)
// Output Y: (B, C_out, H_out=H_in-K+1, W_out=H_out)
//
// All tensors are NCHW row-major flat.  We pass them through the Kuiper-
// extracted kernel [Kuiper_KB_Conv2DSquare_conv2d_square_f32] which
// implements naïve direct convolution (one thread per output element,
// loop over Cin*K*K taps).  The verified entry point requires a bias
// array; for bias=False we allocate a [C_out]-sized buffer and zero
// it with cudaMemset before invocation.
#include <torch/extension.h>
#include "Kuiper_KB_Conv2DSquare.h"
#include "Kuiper_KB_Conv2DSquare.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_conv2d_square_cuda(torch::Tensor X, torch::Tensor W) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda(),
                "kuiper_conv2d_square: all tensors must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                W.scalar_type() == torch::kFloat32,
                "kuiper_conv2d_square: all tensors must be float32");
    TORCH_CHECK(X.dim() == 4 && W.dim() == 4,
                "kuiper_conv2d_square: X and W must be 4D");
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
    TORCH_CHECK(Hin == Win, "kuiper_conv2d_square: H_in must equal W_in");
    TORCH_CHECK(Kh == Kw,   "kuiper_conv2d_square: weight must be square");
    TORCH_CHECK(Cin == WCin,"kuiper_conv2d_square: Cin mismatch");
    TORCH_CHECK(Hin >= Kh,  "kuiper_conv2d_square: H_in < K");
    TORCH_CHECK(Hin <= (int64_t)UINT32_MAX && Kh <= (int64_t)UINT32_MAX,
                "kuiper_conv2d_square: output-size arguments out of u32 range");
    int64_t Hout = (int64_t)Kuiper_KB_Conv2DSquare_conv2d_square_out_sz(
        (uint32_t)Hin, (uint32_t)Kh);
    int64_t nthr = B * Cout * Hout * Hout;
    int64_t xnel = B * Cin * Hin * Hin;
    int64_t wnel = Cout * Cin * Kh * Kh;
    int64_t ynel = nthr;
    TORCH_CHECK(B > 0 && Cin > 0 && Hin > 0 && Cout > 0 && Kh > 0 && Hout > 0,
                "kuiper_conv2d_square: shapes must be positive");
    TORCH_CHECK(B    <= (int64_t)UINT32_MAX &&
                Cin  <= (int64_t)UINT32_MAX &&
                Hin  <= (int64_t)UINT32_MAX &&
                Cout <= (int64_t)UINT32_MAX &&
                Kh   <= (int64_t)UINT32_MAX &&
                Hout <= (int64_t)UINT32_MAX &&
                xnel <= (int64_t)UINT32_MAX &&
                wnel <= (int64_t)UINT32_MAX &&
                ynel <= (int64_t)UINT32_MAX &&
                nthr <= KUIPER_MAX_NTHR,
                "kuiper_conv2d_square: shape out of range");

    auto Y = torch::empty({B, Cout, Hout, Hout}, Xc.options());

    // bias=False ⇒ allocate a zeroed cout-sized buffer; the verified
    // entry point requires a bias array.
    float *bias_d = nullptr;
    size_t bias_bytes = (size_t)Cout * sizeof(float);
    TORCH_CHECK(cudaMalloc(&bias_d, bias_bytes) == cudaSuccess,
                "kuiper_conv2d_square: cudaMalloc bias failed");
    TORCH_CHECK(cudaMemset(bias_d, 0, bias_bytes) == cudaSuccess,
                "kuiper_conv2d_square: cudaMemset bias failed");

    Kuiper_KB_Conv2DSquare_conv2d_square_f32(
        (uint32_t)B, (uint32_t)Cin, (uint32_t)Hin,
        (uint32_t)Cout, (uint32_t)Kh, (uint32_t)Hout,
        Xc.data_ptr<float>(), Wc.data_ptr<float>(),
        bias_d, Y.data_ptr<float>());

    cudaFree(bias_d);
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_conv2d_square", &kuiper_conv2d_square_cuda,
          "Kuiper verified Conv2D forward (square kernel, stride=1, pad=0, "
          "dilation=1, bias=False)");
}
