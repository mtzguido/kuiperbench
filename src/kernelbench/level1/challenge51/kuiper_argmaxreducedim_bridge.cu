// Bridge for KernelBench L1 #51: argmax over the middle dim of a
// (B, D, M) float32 tensor.
//
// PyTorch: y = torch.argmax(x, dim=1), shape (B, M), dtype int64.
//
// Kuiper: factor (B, D, M) row-major as Array2 with layout
//     l2_bcm_pages B M D
// whose imap (r, ci) -> (r/M)*D*M + ci*M + r%M  matches the physical
// row-major (B, D, M) layout.  Row r = b*M + j carries the length-D
// slice x[b,:,j].  One launch of [reduce_batched_argmax_f32] produces
// y[b*M+j] = an i64 index k such that x[b, k, j] == max_{k'} x[b, k', j].
// (Strict-greater-than update gives "first occurrence" runtime
// semantics matching torch.argmax; the Kuiper proof verifies the full
// first-occurrence property — see skeptic.txt.)
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "Kuiper_KB_ArgmaxReduceDim.h"

#include "Kuiper_KB_ArgmaxReduceDim.cu"

// max_blocks * max_threads from Kuiper.Base = 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_argmaxreduce_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 3,
                "kuiper_argmaxreduce_dim1: expected 3-D float32 CUDA tensor");
    TORCH_CHECK(X.is_contiguous(),
                "kuiper_argmaxreduce_dim1: input must be contiguous");
    int64_t B = X.size(0), D = X.size(1), M = X.size(2);
    TORCH_CHECK(B > 0 && D > 0 && M > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && M <= (int64_t)UINT32_MAX
                && (__int128) B * M <= UINT32_MAX
                && (__int128) M * D <= UINT32_MAX
                && (__int128) B * M * D <= UINT32_MAX
                && (__int128) B * M <=
                       (__int128) KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS,
                "kuiper_argmaxreduce_dim1: shape out of range");
    const c10::cuda::CUDAGuard device_guard(X.device());
    int64_t *out = Kuiper_KB_ArgmaxReduceDim_argmaxreduce_dim_alloc_f32(
        (uint32_t)B, (uint32_t)M, (uint32_t)D,
        X.data_ptr<float>());
    return torch::from_blob(out, {B, M},
                            [](void *p) { cudaFree(p); },
                            X.options().dtype(torch::kInt64));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_argmaxreduce_dim1", &kuiper_argmaxreduce_dim1_cuda,
          "Kuiper verified argmax reduction over dim=1 of a 3-D tensor");
}
