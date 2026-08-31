// Bridge for KernelBench L1 #37: FrobeniusNorm.  Mutates input in place.
#include <torch/extension.h>
#include "Kuiper_KB_Frobenius.h"
#include "Kuiper_KB_Frobenius.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_frobenius_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32,
                "kuiper_frobenius: expected CUDA float32 tensor");
    auto Xc = X.contiguous();
    int64_t numel = Xc.numel();
    TORCH_CHECK(numel > 0 && numel <= KUIPER_MAX_NTHR,
                "kuiper_frobenius: numel exceeds verified launch range");
    Kuiper_KB_Frobenius_frobenius_fw_f32((uint32_t)numel, Xc.data_ptr<float>());
    return Xc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_frobenius", &kuiper_frobenius_cuda,
          "Kuiper verified Frobenius norm normalization");
}
