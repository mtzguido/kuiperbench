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
// y[b*M+j] = max_k x[b,k,j] (bit-exact, since IEEE-754 fmax is
// associative+commutative on the modeled carrier).
#include <torch/extension.h>
#include <cmath>
#include "Kuiper_KB_MaxReduceDim.h"

// Rename the [neg_inf] axiom from Kuiper.Math.Fmax to a name we can
// safely supply.  The Python wrapper has stripped the corresponding
// host [extern float ...] declaration from the .cu, so the only
// remaining declaration of [Kuiper_Math_Fmax_neg_inf] is the kernel
// body's read, which we redirect via the preprocessor to a
// __device__ float that we define here.
//
// IEEE-754 [-INFINITY] is the unique f32 value satisfying
// [is_neutral_for neg_inf fmaxf] required by Kuiper.Math.Fmax.
#define Kuiper_Math_Fmax_neg_inf _kuiper_neg_inf_dev
__device__ float _kuiper_neg_inf_dev = -INFINITY;

#include "Kuiper_KB_MaxReduceDim.cu"

#undef Kuiper_Math_Fmax_neg_inf

// max_blocks * max_threads from Kuiper.Base = 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_maxreduce_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 3,
                "kuiper_maxreduce_dim1: expected 3-D float32 CUDA tensor");
    auto Xc = X.contiguous();
    int64_t B = Xc.size(0), D = Xc.size(1), M = Xc.size(2);
    TORCH_CHECK(B > 0 && D > 0 && M > 0
                && B <= (int64_t)UINT32_MAX
                && D <= (int64_t)UINT32_MAX
                && M <= (int64_t)UINT32_MAX
                && B * M <= (int64_t)UINT32_MAX
                && M * D <= (int64_t)UINT32_MAX
                && B * M * D <= (int64_t)UINT32_MAX
                && B * M <= KUIPER_MAX_BLOCKS * KUIPER_MAX_THREADS,
                "kuiper_maxreduce_dim1: shape out of range");
    auto Y = torch::empty({B, M}, Xc.options());
    Kuiper_KB_MaxReduceDim_maxreduce_dim_fw_f32(
        (uint32_t)B, (uint32_t)M, (uint32_t)D,
        Xc.data_ptr<float>(), Y.data_ptr<float>());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_maxreduce_dim1", &kuiper_maxreduce_dim1_cuda,
          "Kuiper verified max reduction over dim=1 of a 3-D tensor");
}
