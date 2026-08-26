// Bridge for KernelBench L1 #53: min reduction over the middle dim of
// a (B, D, M) tensor.
//
// PyTorch: y = torch.min(x, dim=1).values, shape (B, M).
//
// Kuiper: factor (B, D, M) row-major as Array2 with layout
//     l2_bcm_pages B M D
// whose imap (r, ci) -> (r/M)*D*M + ci*M + r%M  matches the physical
// row-major (B, D, M) layout.  Row r = b*M + j carries the length-D
// slice x[b,:,j].  One launch of [reduce_batched_min_f32] produces
// y[b*M+j] = min_k x[b,k,j] (bit-exact, since IEEE-754 fmin is
// associative+commutative on the modeled carrier).
#include <torch/extension.h>
#include <cmath>
#include "Kuiper_KB_MinReduceDim.h"

// Rename the [pos_inf] axiom from Kuiper.Math.Fmin to a name we can
// safely supply.  The Python wrapper has stripped the corresponding
// host [extern float ...] declaration from the .cu, so the only
// remaining declaration of [Kuiper_Math_Fmin_pos_inf] is the kernel
// body's read, which we redirect via the preprocessor to a
// __device__ float that we define here.
//
// IEEE-754 [INFINITY] is the unique f32 value satisfying
// [is_neutral_for pos_inf fminf] required by Kuiper.Math.Fmin.
#define Kuiper_Math_Fmin_pos_inf _kuiper_pos_inf_dev
__device__ float _kuiper_pos_inf_dev = INFINITY;

#include "Kuiper_KB_MinReduceDim.cu"

#undef Kuiper_Math_Fmin_pos_inf

// max_blocks * max_threads from Kuiper.Base = 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_BLOCKS  = (int64_t)2097152;
static constexpr int64_t KUIPER_MAX_THREADS = (int64_t)1024;

torch::Tensor kuiper_minreduce_dim1_cuda(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.dim() == 3,
                "kuiper_minreduce_dim1: expected 3-D float32 CUDA tensor");
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
                "kuiper_minreduce_dim1: shape out of range");
    auto Y = torch::empty({B, M}, Xc.options());
    Kuiper_KB_MinReduceDim_minreduce_dim_fw_f32(
        (uint32_t)B, (uint32_t)M, (uint32_t)D,
        Xc.data_ptr<float>(), Y.data_ptr<float>());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_minreduce_dim1", &kuiper_minreduce_dim1_cuda,
          "Kuiper verified min reduction over dim=1 of a 3-D tensor");
}
