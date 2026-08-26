// Bridge for KernelBench L1 #37: FrobeniusNorm.  Mutates input in place.
#include <torch/extension.h>
#include "Kuiper_KB_Frobenius.h"
#include "Kuiper_KB_Frobenius.cu"

torch::Tensor kuiper_frobenius_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32,
                "kuiper_frobenius: expected CUDA float32 tensor");
    auto Xc = X.contiguous();
    int64_t numel = Xc.numel();
    TORCH_CHECK(numel > 0 && numel <= (int64_t)UINT32_MAX,
                "kuiper_frobenius: numel out of uint32 range");
    Kuiper_KB_Frobenius_frobenius_fw_f32((uint32_t)numel, Xc.data_ptr<float>());
    return Xc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_frobenius", &kuiper_frobenius_cuda,
          "Kuiper verified Frobenius norm normalization");
}
