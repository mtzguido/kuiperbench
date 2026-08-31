// Bridge for KernelBench L1 #41: MaxPool1D.
//
// PyTorch's nn.MaxPool1d(K, S, P, D) takes input (B, C, L) and produces
// (B, C, L_out) with L_out = floor((L + 2P - D*(K-1) - 1)/S) + 1.  Padding
// is implicit -inf for max-pooling.
//
// The verified, self-allocating entry point
// [Kuiper_KB_MaxPool1D_maxpool1d_alloc_f32] takes ONLY the raw (B,C,L,K,S,P,D)
// dims and the input buffer, and *inside the verification boundary*:
//   * computes L_out via the verified [pool_out_len_1d_sz],
//   * allocates the (B*C, L_out) GPU output buffer (extracts to cudaMalloc),
//   * flattens (B,C) -> bc = B*C and runs windowreduce with the f32 fmax
//     monoid (rid = -inf, rop = fmaxf),
//   * returns the pair (L_out, output_device_ptr).
// Ownership of the returned buffer passes to this bridge, which wraps it in a
// torch::Tensor with a cudaFree deleter.  This driver therefore performs NO
// arithmetic and NO allocation that feeds the kernel; it only checks
// dimension contracts on the raw dims (the verified preconditions).
#include <torch/extension.h>
#include <cuda_runtime.h>

// Karamel emits the verified entry's return value as a Prims dtuple2 struct,
// but the build drops the Prims namespace (-drop Prims), so the struct's
// definition is not present in the generated header.  Declare the ABI type
// here (matching Karamel's layout: { .fst = L_out, .snd = device ptr }) before
// including the extracted code.  This is a pure ABI declaration -- no kernel
// logic lives here.
typedef struct Prims_dtuple2__uint32_t__float__s {
    uint32_t fst;
    float *snd;
} Prims_dtuple2__uint32_t__float_;

// Karamel's compound-literal compatibility macro (emitted by the gpu-branch
// krml but absent from this karamel runtime header).  nvcc compiles as C++,
// where `T { ... }` is valid aggregate initialization.
#ifndef KRML_CLITERAL
#  if defined(__cplusplus)
#    define KRML_CLITERAL(x) x
#  else
#    define KRML_CLITERAL(x) (x)
#  endif
#endif

#include "Kuiper_KB_MaxPool1D.h"
#include "Kuiper_KB_MaxPool1D.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

torch::Tensor kuiper_maxpool1d_cuda(torch::Tensor X,
                                    int64_t kernel_size,
                                    int64_t stride,
                                    int64_t padding,
                                    int64_t dilation) {
    TORCH_CHECK(X.is_cuda(), "kuiper_maxpool1d: X must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
                "kuiper_maxpool1d: X must be float32");
    TORCH_CHECK(X.dim() == 3, "kuiper_maxpool1d: X must be (B, C, L)");
    TORCH_CHECK(kernel_size >= 1 && stride >= 1 && padding >= 1
                && dilation >= 1,
                "kuiper_maxpool1d: k/s/p/d must be >= 1 (verified szp range)");

    auto Xc = X.contiguous();
    int64_t B = Xc.size(0);
    int64_t C = Xc.size(1);
    int64_t L = Xc.size(2);

    // Raw-dimension contract checks discharging the verified preconditions of
    // [maxpool1d_alloc_f32].  No L_out is computed here: the kernel computes,
    // allocates, fills, and returns it.  [kspan = D*(K-1)+1], [padded = L+2P].
    int64_t kspan  = dilation * (kernel_size - 1) + 1;
    int64_t padded = L + 2 * padding;
    TORCH_CHECK(B > 0 && C > 0 && L > 0
                && B * C > 0 && B * C <= (int64_t)UINT32_MAX
                && L <= (int64_t)UINT32_MAX
                && kspan <= (int64_t)UINT32_MAX
                && padded <= (int64_t)UINT32_MAX
                && kspan <= padded                       /* window fits => L_out > 0 */
                && padded * stride + kernel_size * dilation <= (int64_t)UINT32_MAX
                && B * C * L <= (int64_t)UINT32_MAX
                && B * C * padded <= (int64_t)UINT32_MAX
                && B * C * padded <= KUIPER_MAX_NTHR,
                "kuiper_maxpool1d: shape out of verified u32 / launch range");

    Prims_dtuple2__uint32_t__float_ r =
        Kuiper_KB_MaxPool1D_maxpool1d_alloc_f32(
            (uint32_t)B, (uint32_t)C, (uint32_t)L,
            (uint32_t)kernel_size, (uint32_t)stride,
            (uint32_t)padding,     (uint32_t)dilation,
            Xc.data_ptr<float>());

    int64_t L_out = (int64_t)r.fst;
    float *out_ptr = r.snd;

    // Wrap the Kuiper-allocated (cudaMalloc'd) device buffer in a tensor that
    // owns it: the deleter cudaFree's it when the tensor is destroyed.
    auto Y = torch::from_blob(
        out_ptr, {B, C, L_out},
        [](void *p) { cudaFree(p); },
        Xc.options());
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_maxpool1d", &kuiper_maxpool1d_cuda,
          "Kuiper verified MaxPool1D (self-allocating windowreduce, fmax monoid)");
}
