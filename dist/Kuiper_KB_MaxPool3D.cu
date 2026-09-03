
#include "Kuiper_KB_MaxPool3D.h"

uint32_t Kuiper_KB_MaxPool3D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
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
  hoisted when extracting maxpool3d_axis_fw_rm_f32
*/
static void
__hoisted_maxpool3d_axis_fw_rm_f32_0(uint32_t k, uint32_t s, uint32_t p,
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

void Kuiper_KB_MaxPool3D_maxpool3d_axis_fw_rm_f32(uint32_t k, uint32_t s,
                                                  uint32_t p, uint32_t d,
                                                  uint32_t bc, uint32_t l,
                                                  uint32_t l_out, float *input,
                                                  float *output)
{
    if (l_out != 0U) {
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_maxpool3d_axis_fw_rm_f32_0,
                  bc * l_out / 1024U + (uint32_t) (bc * l_out % 1024U != 0U),
                  1024U, 0U, s1, k, s, p, d, bc, l, l_out, input, output);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
    }
}

__global__
/**
  hoisted when extracting maxpool3d_raw_alloc_f32
*/
static void
__hoisted_maxpool3d_raw_alloc_f32_0(uint32_t k, uint32_t s, uint32_t p,
                                    uint32_t d, uint32_t h, uint32_t wo,
                                    float *mid_h_in, uint32_t ho,
                                    uint32_t rows_h, float *mid_h)
{
    if (1024U * blockIdx.x + threadIdx.x < rows_h * ho) {
        uint32_t r_sz = (1024U * blockIdx.x + threadIdx.x) / ho;
        uint32_t j_sz = (1024U * blockIdx.x + threadIdx.x) % ho;
        float acc = -INFINITY;
        uint32_t di_ref = 0U;
        float buf = -INFINITY;
        KRML_HOST_IGNORE(&buf);
        for (; di_ref < k; di_ref++) {
            uint32_t pos = j_sz * s + di_ref * d;
            bool in_bounds = pos >= p && pos - p < h;
            uint32_t dpos_ref = 0U;
            if (in_bounds)
                dpos_ref = pos - p;
            float raw =
                mid_h_in[r_sz / wo * h * wo + dpos_ref * wo + r_sz % wo];
            acc = fmaxf(acc, in_bounds ? raw : -INFINITY);
        }
        mid_h[r_sz / wo * ho * wo + j_sz * wo + r_sz % wo] = acc;
    }
}

__global__
/**
  hoisted when extracting maxpool3d_raw_alloc_f32
*/
static void
__hoisted_maxpool3d_raw_alloc_f32_1(uint32_t k, uint32_t s, uint32_t p,
                                    uint32_t d, uint32_t depth, uint32_t wo,
                                    uint32_t ho, float *mid_d_in, uint32_t do_,
                                    uint32_t rows_d, float *out)
{
    if (1024U * blockIdx.x + threadIdx.x < rows_d * do_) {
        uint32_t r_sz = (1024U * blockIdx.x + threadIdx.x) / do_;
        uint32_t j_sz = (1024U * blockIdx.x + threadIdx.x) % do_;
        float acc = -INFINITY;
        uint32_t di_ref = 0U;
        float buf = -INFINITY;
        KRML_HOST_IGNORE(&buf);
        for (; di_ref < k; di_ref++) {
            uint32_t pos = j_sz * s + di_ref * d;
            bool in_bounds = pos >= p && pos - p < depth;
            uint32_t dpos_ref = 0U;
            if (in_bounds)
                dpos_ref = pos - p;
            float raw = mid_d_in[r_sz / (ho * wo) * depth * ho * wo +
                                 dpos_ref * ho * wo + r_sz % (ho * wo)];
            acc = fmaxf(acc, in_bounds ? raw : -INFINITY);
        }
        out[r_sz / (ho * wo) * do_ * ho * wo + j_sz * ho * wo +
            r_sz % (ho * wo)] = acc;
    }
}

Kuiper_KB_MaxPool3D_maxpool3d_full_result
Kuiper_KB_MaxPool3D_maxpool3d_raw_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                            uint32_t d, uint32_t b, uint32_t c,
                                            uint32_t depth, uint32_t h,
                                            uint32_t w, float *input)
{
    uint32_t wo = Kuiper_KB_MaxPool3D_pool_out_len_1d_sz(w, k, s, p, d);
    uint32_t rows_w = b * c * depth * h;
    float *mid_w = (float *) KPR_GPU_ALLOC(sizeof(float), rows_w * wo);
    Kuiper_KB_MaxPool3D_maxpool3d_axis_fw_rm_f32(k, s, p, d, rows_w, w, wo,
                                                 input, mid_w);
    float *mid_h_in = mid_w;
    uint32_t ho = Kuiper_KB_MaxPool3D_pool_out_len_1d_sz(h, k, s, p, d);
    uint32_t rows_h = b * c * depth * wo;
    float *mid_h = (float *) KPR_GPU_ALLOC(sizeof(float), rows_h * ho);
    if (ho != 0U) {
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_maxpool3d_raw_alloc_f32_0,
                  rows_h * ho / 1024U + (uint32_t) (rows_h * ho % 1024U != 0U),
                  1024U, 0U, s1, k, s, p, d, h, wo, mid_h_in, ho, rows_h,
                  mid_h);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
    }
    MUST(cudaFree(mid_h_in));
    float *mid_d_in = mid_h;
    uint32_t do_ = Kuiper_KB_MaxPool3D_pool_out_len_1d_sz(depth, k, s, p, d);
    uint32_t rows_d = b * c * ho * wo;
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), rows_d * do_);
    if (do_ != 0U) {
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_maxpool3d_raw_alloc_f32_1,
                  rows_d * do_ / 1024U +
                      (uint32_t) (rows_d * do_ % 1024U != 0U),
                  1024U, 0U, s1, k, s, p, d, depth, wo, ho, mid_d_in, do_,
                  rows_d, out);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
    }
    MUST(cudaFree(mid_d_in));
    return (KRML_CLITERAL(Kuiper_KB_MaxPool3D_maxpool3d_full_result){
        .w_out = wo, .h_out = ho, .d_out = do_, .output = out});
}
