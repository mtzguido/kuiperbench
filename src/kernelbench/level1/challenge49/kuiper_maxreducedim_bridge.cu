// Bridge for KernelBench L1 #49: max reduction over the middle dim of
// a (B, D, M) tensor.
//
// PyTorch: y = torch.max(x, dim=1).values, shape (B, M).
//
// Kuiper: factor (B, D, M) row-major as Array2 with layout
//     l2_bcm_pages B M D
// whose imap (r, ci) -> (r/M)*D*M + ci*M + r%M  matches the physical
// row-major (B, D, M) layout.  Row r = b*M + j carries the length-D
// slice x[b,:,j].  One launch of [reduce_batched_max_f32] produces
// y[b*M+j] is the exact deterministic fmax reduction specified by Kuiper;
// the proof does not assume blanket fmax algebraic laws over NaNs.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "Kuiper_KB_MaxReduceDim.h"
#include "Kuiper_KB_MaxReduceDim.cu"

// max_blocks * max_threads from Kuiper.Base = 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_maxreduce_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 3,
                "kuiper_maxreduce_dim1: expected 3-D float32 CUDA tensor");
    TORCH_CHECK(X.is_contiguous(),
                "kuiper_maxreduce_dim1: input must be contiguous");
    int64_t B = X.size(0), D = X.size(1), M = X.size(2);
    const __int128 BM = (__int128)B * (__int128)M;
    const __int128 MD = (__int128)M * (__int128)D;
    const __int128 BMD = BM * (__int128)D;
    const __int128 D_with_threads =
        (__int128)D + (__int128)KUIPER_MAX_THREADS;
    TORCH_CHECK(B > 0 && D > 0 && M > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && M <= (int64_t)UINT32_MAX
                && BM <= (__int128)UINT32_MAX
                && MD <= (__int128)UINT32_MAX
                && BMD <= (__int128)UINT32_MAX
                && BM <= (__int128)KUIPER_MAX_BLOCKS
                && D_with_threads <= (__int128)UINT32_MAX,
                "kuiper_maxreduce_dim1: shape out of range");
    const c10::cuda::CUDAGuard device_guard(X.device());
    float *out = Kuiper_KB_MaxReduceDim_maxreduce_dim_alloc_f32(
        (uint32_t)B, (uint32_t)M, (uint32_t)D,
        X.data_ptr<float>());
    return torch::from_blob(out, {B, M},
                            [](void *p) { cudaFree(p); }, X.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_maxreduce_dim1", &kuiper_maxreduce_dim1_cuda,
          "Kuiper verified max reduction over dim=1 of a 3-D tensor");
}
