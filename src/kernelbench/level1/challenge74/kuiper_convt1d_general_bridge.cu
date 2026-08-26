// L1 #64 bridge: ConvTranspose1D, implemented by reusing the verified
// ConvT2D kernel with H=Kh=1.  Input (B,Cin,L)→(B,Cin,1,L); weight
// (Cin,Cout,K)→(Cin,Cout,1,K); output (B,Cout,L_out)→(B,Cout,1,L_out).
#include "kuiper_convt2d_general_common.h"

static torch::Tensor kuiper_convt1d_general_cuda(
        torch::Tensor X, torch::Tensor W,
        c10::optional<torch::Tensor> Bias_opt,
        int64_t stride, int64_t padding, int64_t output_padding,
        int64_t dilation, int64_t groups) {
    TORCH_CHECK(X.dim() == 3 && W.dim() == 3,
                "kuiper_convt1d_general: X and W must be 3D");
    auto Xc = X.contiguous(); auto Wc = W.contiguous();
    // Reshape to 4D with H=1 / Kh=1; weight is (Cin, Cout, K).
    auto X4 = Xc.unsqueeze(2);  // (B, Cin, 1, L)
    auto W4 = Wc.unsqueeze(2);  // (Cin, Cout, 1, K)
    auto Y4 = kuiper_convt2d_general_cuda(
        X4, W4, Bias_opt,
        /*sh*/ 1, /*sw*/ stride,
        /*ph*/ 0, /*pw*/ padding,
        /*oph*/ 0, /*opw*/ output_padding,
        /*dh*/ 1, /*dw*/ dilation,
        groups);
    return Y4.squeeze(2);  // (B, Cout, L_out)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_convt1d_general", &kuiper_convt1d_general_cuda,
          "Kuiper verified ConvTranspose1D forward (general parameters)");
}
