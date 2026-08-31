// Bridge for KernelBench L1 #42: MaxPool2D.
//
// PyTorch's nn.MaxPool2d(K, S, P, D) takes (B, C, H, W) and produces
// (B, C, H_out, W_out) with H_out/W_out = floor((dim + 2P - D*(K-1) - 1)/S) + 1.
// Padding contributes implicit -inf to the max.
//
// MaxPool is separable: max{x[h+dh, w+dw]} = max_dh max_dw x[h+dh, w+dw].
// The ENTIRE 2-D pool is now a SINGLE verified F*/Pulse call
// [Kuiper_KB_MaxPool2D_maxpool2d_full_alloc_f32]: it takes ONLY raw
// (B*C, H, W, kh,kw,sh,sw,ph,pw,dh,dw) and the input buffer, and *inside the
// verification boundary*:
//
//   pass 1: view input as (B*C*H, W); reduce over W ⇒ row-major (B*C, H, W_out)
//   pass 2: reinterpret the SAME bytes as the flat batched-column-major
//           (l2_bcm_pages) view (B*C*W_out, H) and reduce the now-strided H
//           axis directly into a freshly allocated row-major (B*C, H_out, W_out)
//           buffer.
//
// There is NO physical transpose: the inter-pass permute+contiguous copies
// that used to live in this bridge (and were NOT verified by Kuiper) are gone.
// The call allocates the intermediate, frees it after pass 2, and returns
// (W_out, (H_out, device_ptr)) where device_ptr is already the row-major
// (B, C, H_out, W_out) result.  This bridge does NO arithmetic that feeds the
// kernel and NO output allocation; it only checks dimension contracts.
#include <torch/extension.h>
#include <cuda_runtime.h>

// Karamel emits the verified entry's return value as nested Prims dtuple2
// structs, but the build drops the Prims namespace (-drop Prims), so their
// definitions are not present in the generated header.  Declare the ABI types
// here (matching Karamel's layout) before including the extracted code.  Pure
// ABI declarations -- no kernel logic.
typedef struct Prims_dtuple2__uint32_t__float__s {
    uint32_t fst;
    float *snd;
} Prims_dtuple2__uint32_t__float_;

typedef struct Prims_dtuple2__uint32_t_Prims_dtuple2__uint32_t__float__s {
    uint32_t fst;
    Prims_dtuple2__uint32_t__float_ snd;
} Prims_dtuple2__uint32_t_Prims_dtuple2__uint32_t__float_;

// Karamel's compound-literal compatibility macro (emitted by the gpu-branch
// krml but absent from this karamel runtime header).
#ifndef KRML_CLITERAL
#  if defined(__cplusplus)
#    define KRML_CLITERAL(x) x
#  else
#    define KRML_CLITERAL(x) (x)
#  endif
#endif

// The extracted entry projects the inner pass-1 dtuple with FStar.Pervasives's
// generic dfst/dsnd, whose monomorphic definitions live in the dropped Prims/
// FStar namespaces.  They are plain struct-field projections; provide them as
// macros (pure ABI shims, no kernel logic) before including the extracted code.
#ifndef FStar_Pervasives_dfst
#  define FStar_Pervasives_dfst(x) ((x).fst)
#endif
#ifndef FStar_Pervasives_dsnd
#  define FStar_Pervasives_dsnd(x) ((x).snd)
#endif

#include "Kuiper_KB_MaxPool2D.h"
#include "Kuiper_KB_MaxPool2D.cu"

// max_blocks * max_threads from Kuiper.Base: 2^21 * 1024 = 2^31
static constexpr int64_t KUIPER_MAX_NTHR = (int64_t)2097152 * 1024;

// Raw-dimension contract checks discharging the verified preconditions of
// [maxpool2d_full_alloc_f32].  No W_out/H_out is computed here: the kernel
// computes, allocates, fills, frees, and returns them.
static void check_full_raw(int64_t bc, int64_t H, int64_t W,
                           int64_t k, int64_t s, int64_t p, int64_t d) {
    int64_t kspanW = d * (k - 1) + 1;
    int64_t kspanH = d * (k - 1) + 1;
    int64_t pw = W + 2 * p;
    int64_t ph = H + 2 * p;
    int64_t U = (int64_t)UINT32_MAX;
    TORCH_CHECK(bc > 0
                && H <= U && W <= U
                && kspanW <= U && kspanH <= U
                && pw <= U && ph <= U
                && kspanW <= pw                          /* W window fits => W_out > 0 */
                && kspanH <= ph                          /* H window fits => H_out > 0 */
                && pw * s + k * d <= U
                && ph * s + k * d <= U
                && bc * ph * pw <= U                      /* fits(bc*(H+2p)*(W+2p)) */
                && bc * ph * pw <= KUIPER_MAX_NTHR,
                "kuiper_maxpool2d: shape out of verified u32 / launch range");
}

torch::Tensor kuiper_maxpool2d_cuda(torch::Tensor X,
                                    int64_t kernel_size,
                                    int64_t stride,
                                    int64_t padding,
                                    int64_t dilation) {
    TORCH_CHECK(X.is_cuda(), "kuiper_maxpool2d: X must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32,
                "kuiper_maxpool2d: X must be float32");
    TORCH_CHECK(X.dim() == 4, "kuiper_maxpool2d: X must be (B, C, H, W)");
    TORCH_CHECK(kernel_size >= 1 && stride >= 1 && padding >= 1
                && dilation >= 1,
                "kuiper_maxpool2d: k/s/p/d must be >= 1 (verified szp range)");

    auto Xc = X.contiguous();
    int64_t B = Xc.size(0);
    int64_t C = Xc.size(1);
    int64_t H = Xc.size(2);
    int64_t W = Xc.size(3);
    int64_t bc = B * C;

    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0,
                "kuiper_maxpool2d: input dimensions must be positive");
    check_full_raw(bc, H, W, kernel_size, stride, padding, dilation);

    // ── Single verified, transpose-free 2-D max pool. ────────────────
    // The same scalar (k,s,p,d) is used for both axes: kh=kw, sh=sw, etc.
    Prims_dtuple2__uint32_t_Prims_dtuple2__uint32_t__float_ r =
        Kuiper_KB_MaxPool2D_maxpool2d_full_alloc_f32(
            (uint32_t)kernel_size, (uint32_t)kernel_size,   // kh, kw
            (uint32_t)stride,      (uint32_t)stride,        // sh, sw
            (uint32_t)padding,     (uint32_t)padding,       // ph, pw
            (uint32_t)dilation,    (uint32_t)dilation,      // dh, dw
            (uint32_t)bc, (uint32_t)H, (uint32_t)W,
            Xc.data_ptr<float>());

    int64_t W_out = (int64_t)r.fst;
    int64_t H_out = (int64_t)r.snd.fst;
    float *out_ptr = r.snd.snd;

    // out_ptr is already the row-major (B, C, H_out, W_out) result -- no
    // permute needed.  Wrap it in a tensor owning the cudaMalloc'd buffer.
    return torch::from_blob(
        out_ptr, {B, C, H_out, W_out},
        [](void *q) { cudaFree(q); }, Xc.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kuiper_maxpool2d", &kuiper_maxpool2d_cuda,
          "Kuiper verified MaxPool2D (single transpose-free windowreduce entry)");
}
