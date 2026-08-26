// Bridge for KernelBench L1 #45: AvgPool2D.
//
// PyTorch's nn.AvgPool2d(K, S, P) takes (B, C, H, W) and produces
// (B, C, H_out, W_out).  Default count_include_pad=True so the divisor
// is K*K regardless of how much of the window lies in padding.  Padding
// contributes 0 to the sum.  KB harness uses K=11, default S=K, P=0.
//
// Avg-pool sum is separable: sum_(dh,dw) x[h+dh, w+dw] = sum_dh sum_dw,
// and dividing by K after EACH axis pass gives /(K*K) total.  Each pass
// is realised by the SINGLE verified, self-allocating per-axis entry
// [Kuiper_KB_AvgPool2D_avgpool2d_axis_alloc_f32], which *inside the
// verification boundary*:
//   * computes L_out via the verified [pool_out_len_1d_sz],
//   * allocates the (bc, L_out) GPU output buffer (cudaMalloc),
//   * fills it with the per-window SUM (cmonoid_fadd_f32; rid=0, rop=+),
//   * divides every element by K in place (verified ScalarMul, *1/K), and
//   * returns the pair (L_out, output_device_ptr).
// The /K that DEFINES average pooling is therefore VERIFIED, not done in
// this driver.  The driver performs NO arithmetic on tensor data, NO
// allocation, and NO separate scale launch; it only:
//   * checks the raw-dimension contracts (the verified preconditions),
//   * computes L_out via the verified pool_out_len_1d_sz (for buffer
//     reshaping only),
//   * chunks the leading row dimension when it would overflow the
//     verified primitive's u32 size bound (the KB shape has
//     B*C*H*W = 2^32 which overflows by 1; see skeptic.txt gap #4),
//   * stitches per-chunk buffers with torch::cat (a layout op, like the
//     inter-pass permute), and
//   * does the inter-pass permute+contiguous (no verified transpose yet).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

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

#include "Kuiper_KB_AvgPool2D.h"
#include "Kuiper_KB_AvgPool2D.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

// Run ONE separable axis pass over a [(total_rows, L)] row-major buffer:
// per-window SUM over the inner axis then /K, both verified inside
// [avgpool2d_axis_alloc_f32].  Returns a fresh [(total_rows, L_out)] tensor
// owning the Kuiper-allocated device buffer(s).  When [total_rows * L] would
// overflow the verified u32 size bound, the leading dimension is chunked and
// the per-chunk device buffers are concatenated (a layout op).
static torch::Tensor run_axis(float *in_ptr, int64_t total_rows, int64_t L,
                              int64_t k, int64_t s, int64_t p, int64_t d,
                              const torch::TensorOptions &opts) {
    // Per-axis (bc-independent) verified-precondition contract checks.
    int64_t kspan  = d * (k - 1) + 1;
    int64_t padded = L + 2 * p;
    TORCH_CHECK(L >= 1 && L <= (int64_t)UINT32_MAX
                && kspan <= (int64_t)UINT32_MAX
                && padded <= (int64_t)UINT32_MAX
                && kspan <= padded,                    // window fits => L_out >= 1
                "kuiper_avgpool2d: axis shape out of verified range");
    // L_out via the VERIFIED formula (no unverified length math here).
    int64_t L_out = (int64_t)Kuiper_KB_AvgPool2D_pool_out_len_1d_sz(
        (uint32_t)L, (uint32_t)k, (uint32_t)s, (uint32_t)p, (uint32_t)d);
    TORCH_CHECK(L_out >= 1, "kuiper_avgpool2d: empty output axis");
    TORCH_CHECK(L_out * s + k * d <= (int64_t)UINT32_MAX,
                "kuiper_avgpool2d: l_out*s + k*d overflows u32");

    // Largest row-chunk respecting the verified per-chunk bounds:
    //   bc*L      <= UINT32_MAX        (input  layout fits u32)
    //   bc*L_out  <= UINT32_MAX        (output layout fits u32)
    //   bc*L_out  <= KUIPER_MAX_NTHR   (verified launch / scale bound)
    int64_t cap_in  = (int64_t)UINT32_MAX / L;
    int64_t cap_out = (int64_t)UINT32_MAX / L_out;
    int64_t cap_thr = KUIPER_MAX_NTHR / L_out;
    int64_t cap = std::min(std::min(cap_in, cap_out), cap_thr);
    TORCH_CHECK(cap > 0, "kuiper_avgpool2d: per-row length exceeds u32");

    std::vector<torch::Tensor> chunks;
    for (int64_t off = 0; off < total_rows; off += cap) {
        int64_t bc = std::min(cap, total_rows - off);
        Prims_dtuple2__uint32_t__float_ r =
            Kuiper_KB_AvgPool2D_avgpool2d_axis_alloc_f32(
                (uint32_t)k, (uint32_t)s, (uint32_t)p, (uint32_t)d,
                (uint32_t)bc, (uint32_t)L, in_ptr + off * L);
        // Wrap the Kuiper-allocated (cudaMalloc'd) buffer; deleter cudaFree's it.
        chunks.push_back(torch::from_blob(
            r.snd, {bc, L_out},
            [](void *q) { cudaFree(q); }, opts));
    }
    return chunks.size() == 1 ? chunks[0] : torch::cat(chunks, 0);
}

torch::Tensor kuiper_avgpool2d_cuda(torch::Tensor X,
                                    int64_t kernel_size,
                                    int64_t stride,
                                    int64_t padding) {
    TORCH_CHECK(X.is_cuda(), "kuiper_avgpool2d: X must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
                "kuiper_avgpool2d: X must be float32");
    TORCH_CHECK(X.dim() == 4, "kuiper_avgpool2d: X must be (B, C, H, W)");
    TORCH_CHECK(kernel_size >= 1 && stride >= 1 && padding >= 0,
                "kuiper_avgpool2d: k/s >= 1, p >= 0");

    auto Xc = X.contiguous();
    int64_t B = Xc.size(0);
    int64_t C = Xc.size(1);
    int64_t H = Xc.size(2);
    int64_t W = Xc.size(3);
    int64_t dilation = 1;  // PyTorch nn.AvgPool2d has no dilation arg
    auto opts = Xc.options();

    // -- Pass 1: sum over W axis (then /K). View as (B*C*H, W). --
    auto Y1 = run_axis(Xc.data_ptr<float>(), B * C * H, W,
                       kernel_size, stride, padding, dilation, opts);
    int64_t W_out = Y1.size(1);
    Y1 = Y1.view({B, C, H, W_out});

    // -- Pass 2: sum over H axis (then /K). --
    auto Y1p = Y1.permute({0, 1, 3, 2}).contiguous();   // (B, C, W_out, H)
    auto Y2 = run_axis(Y1p.data_ptr<float>(), B * C * W_out, H,
                       kernel_size, stride, padding, dilation, opts);
    int64_t H_out = Y2.size(1);
    Y2 = Y2.view({B, C, W_out, H_out});

    return Y2.permute({0, 1, 3, 2}).contiguous();        // (B, C, H_out, W_out)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_avgpool2d", &kuiper_avgpool2d_cuda,
          "Kuiper verified AvgPool2D (self-allocating per-axis: windowreduce "
          "fadd sum + verified ScalarMul /K)");
}
