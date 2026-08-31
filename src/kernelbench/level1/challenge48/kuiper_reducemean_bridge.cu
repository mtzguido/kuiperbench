// Bridge for KernelBench L1 #48: mean reduction over the middle dim of
// a (B, D, M) tensor.
//
// PyTorch: y = torch.mean(x, dim=1), shape (B, M) (keepdim=False).
//
// Pipeline (verified):
//   1. reduce_sum_fw_f32 over (B*M, D) view  → y[b*M+j] %~ Σ_k x[b,k,j]
//   2. smul_fw_f32 with c = 1/D in place    → y[b*M+j] *= 1/D
#include <torch/extension.h>
#include "Kuiper_KB_ReduceMean.h"
#include "Kuiper_KB_ReduceMean.cu"

static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_reducemean_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 3,
                "kuiper_reducemean_dim1: expected 3-D float32 CUDA tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), D = Xc.size(1), M = Xc.size(2);
    TORCH_CHECK(B > 0 && D > 0 && M > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && M <= (int64_t)UINT32_MAX
                && B * M <= (int64_t)UINT32_MAX
                && M * D <= (int64_t)UINT32_MAX
                && B * M * D <= (int64_t)UINT32_MAX
                && B * M <= KUIPER_MAX_BLOCKS
                && D + KUIPER_MAX_THREADS <= (int64_t)UINT32_MAX,
                "kuiper_reducemean_dim1: shape out of range");
    auto Y = torch::empty({B, M}, Xc.options());
    Kuiper_KB_ReduceMean_reduce_mean_fw_f32(
        (uint32_t)B, (uint32_t)M, (uint32_t)D,
        Xc.data_ptr<float>(), Y.data_ptr<float>());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_reducemean_dim1", &kuiper_reducemean_dim1_cuda,
          "Kuiper verified mean reduction over dim=1 of a 3-D tensor");
}
