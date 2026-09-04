
#include "Kuiper_KB_MaxPool2D.h"

uint32_t Kuiper_KB_MaxPool2D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
                                                uint32_t s, uint32_t p,
                                                uint32_t d)
{
    uint32_t kspan = d * (k - 1U) + 1U;
    uint32_t padded = l + 2U * p;
    if (padded < kspan)
        return 0U;
    else
        return (padded - kspan) / s + 1U;
}

__global__
/**
  hoisted when extracting maxpool2d_axis_fw_rm_f32
*/
static void
__hoisted_maxpool2d_axis_fw_rm_f32_0(uint32_t k, uint32_t s, uint32_t p,
                                     uint32_t d, uint32_t bc, uint32_t l,
                                     uint32_t l_out, float *input,
                                     float *output)
{
    if (1024U * blockIdx.x + threadIdx.x < bc * l_out) {
        uint32_t r_sz = (1024U * blockIdx.x + threadIdx.x) / l_out;
        uint32_t j_sz = (1024U * blockIdx.x + threadIdx.x) % l_out;
        float acc = -INFINITY;
        uint32_t di_ref = 0U;
        float buf = -INFINITY;
        KRML_HOST_IGNORE(&buf);
        for (; di_ref < k; di_ref++) {
            uint32_t pos = j_sz * s + di_ref * d;
            bool in_bounds = pos >= p && pos - p < l;
            uint32_t dpos_ref = 0U;
            if (in_bounds)
                dpos_ref = pos - p;
            float raw = input[r_sz * l + dpos_ref];
            acc = fmaxf(acc, in_bounds ? raw : -INFINITY);
        }
        output[r_sz * l_out + j_sz] = acc;
    }
}

void Kuiper_KB_MaxPool2D_maxpool2d_axis_fw_rm_f32(uint32_t k, uint32_t s,
                                                  uint32_t p, uint32_t d,
                                                  uint32_t bc, uint32_t l,
                                                  uint32_t l_out, float *input,
                                                  float *output)
{
    if (l_out != 0U) {
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_maxpool2d_axis_fw_rm_f32_0,
                  bc * l_out / 1024U + (uint32_t) (bc * l_out % 1024U != 0U),
                  1024U, 0U, s1, k, s, p, d, bc, l, l_out, input, output);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
    }
}

__global__
/**
  hoisted when extracting maxpool2d_full_alloc_f32
*/
static void
__hoisted_maxpool2d_full_alloc_f32_0(uint32_t kh, uint32_t sh, uint32_t ph,
                                     uint32_t dh, uint32_t bc, uint32_t h,
                                     uint32_t wo, uint32_t ho, float *mid2,
                                     float *out)
{
    if (1024U * blockIdx.x + threadIdx.x < bc * wo * ho) {
        uint32_t r_sz = (1024U * blockIdx.x + threadIdx.x) / ho;
        uint32_t j_sz = (1024U * blockIdx.x + threadIdx.x) % ho;
        float acc = -INFINITY;
        uint32_t di_ref = 0U;
        float buf = -INFINITY;
        KRML_HOST_IGNORE(&buf);
        for (; di_ref < kh; di_ref++) {
            uint32_t pos = j_sz * sh + di_ref * dh;
            bool in_bounds = pos >= ph && pos - ph < h;
            uint32_t dpos_ref = 0U;
            if (in_bounds)
                dpos_ref = pos - ph;
            float raw = mid2[r_sz / wo * h * wo + dpos_ref * wo + r_sz % wo];
            acc = fmaxf(acc, in_bounds ? raw : -INFINITY);
        }
        out[r_sz / wo * ho * wo + j_sz * wo + r_sz % wo] = acc;
    }
}

Kuiper_KB_MaxPool2D_maxpool2d_full_result
Kuiper_KB_MaxPool2D_maxpool2d_full_alloc_f32(uint32_t kh, uint32_t kw,
                                             uint32_t sh, uint32_t sw,
                                             uint32_t ph, uint32_t pw,
                                             uint32_t dh, uint32_t dw,
                                             uint32_t bc, uint32_t h,
                                             uint32_t w, float *input)
{
    uint32_t wo = Kuiper_KB_MaxPool2D_pool_out_len_1d_sz(w, kw, sw, pw, dw);
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), bc * h * wo);
    Kuiper_KB_MaxPool2D_maxpool2d_axis_fw_rm_f32(kw, sw, pw, dw, bc * h, w, wo,
                                                 input, output);
    float *mid = output;
    uint32_t ho = Kuiper_KB_MaxPool2D_pool_out_len_1d_sz(h, kh, sh, ph, dh);
    float *mid2 = mid;
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), bc * wo * ho);
    if (ho != 0U) {
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_maxpool2d_full_alloc_f32_0,
                  bc * wo * ho / 1024U +
                      (uint32_t) (bc * wo * ho % 1024U != 0U),
                  1024U, 0U, s, kh, sh, ph, dh, bc, h, wo, ho, mid2, out);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
    }
    MUST(cudaFree(mid2));
    return (KRML_CLITERAL(Kuiper_KB_MaxPool2D_maxpool2d_full_result){
        .w_out = wo, .h_out = ho, .output = out});
}

Kuiper_KB_MaxPool2D_maxpool2d_full_result
Kuiper_KB_MaxPool2D_maxpool2d_raw_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                            uint32_t d, uint32_t b, uint32_t c,
                                            uint32_t h, uint32_t w,
                                            float *input)
{
    return Kuiper_KB_MaxPool2D_maxpool2d_full_alloc_f32(k, k, s, s, p, p, d, d,
                                                        b * c, h, w, input);
}
