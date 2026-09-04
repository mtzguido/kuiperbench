
#include "Kuiper_KB_AvgPool3D.h"

__global__
/**
  hoisted when extracting smul_fw_f32
*/
static void
__hoisted_smul_fw_f32_0(float c, uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *= c;
}

static void smul_fw_f32(float c, uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, c,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

uint32_t Kuiper_KB_AvgPool3D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
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

float Kuiper_KB_AvgPool3D_avgpool_recip_f32(uint32_t k)
{
    return 1.0f / (float) (int64_t) (uint64_t) k;
}

__global__
/**
  hoisted when extracting avgpool3d_axis_fw_rm_f32
*/
static void
__hoisted_avgpool3d_axis_fw_rm_f32_0(uint32_t k, uint32_t s, uint32_t p,
                                     uint32_t d, uint32_t bc, uint32_t l,
                                     uint32_t l_out, float *input,
                                     float *output)
{
    if (1024U * blockIdx.x + threadIdx.x < bc * l_out) {
        uint32_t r_sz = (1024U * blockIdx.x + threadIdx.x) / l_out;
        uint32_t j_sz = (1024U * blockIdx.x + threadIdx.x) % l_out;
        float acc = 0.0f;
        uint32_t di_ref = 0U;
        float buf = 0.0f;
        KRML_HOST_IGNORE(&buf);
        for (; di_ref < k; di_ref++) {
            uint32_t pos = j_sz * s + di_ref * d;
            bool in_bounds = pos >= p && pos - p < l;
            uint32_t dpos_ref = 0U;
            if (in_bounds)
                dpos_ref = pos - p;
            float raw = input[r_sz * l + dpos_ref];
            acc += in_bounds ? raw : 0.0f;
        }
        output[r_sz * l_out + j_sz] = acc;
    }
}

void Kuiper_KB_AvgPool3D_avgpool3d_axis_fw_rm_f32(uint32_t k, uint32_t s,
                                                  uint32_t p, uint32_t d,
                                                  uint32_t bc, uint32_t l,
                                                  uint32_t l_out, float *input,
                                                  float *output)
{
    if (l_out != 0U) {
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_avgpool3d_axis_fw_rm_f32_0,
                  bc * l_out / 1024U + (uint32_t) (bc * l_out % 1024U != 0U),
                  1024U, 0U, s1, k, s, p, d, bc, l, l_out, input, output);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
    }
}

Kuiper_KB_AvgPool3D_avgpool3d_axis_alloc_result
Kuiper_KB_AvgPool3D_avgpool3d_axis_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                             uint32_t d, uint32_t bc,
                                             uint32_t l, float *input)
{
    uint32_t l_out = Kuiper_KB_AvgPool3D_pool_out_len_1d_sz(l, k, s, p, d);
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), bc * l_out);
    Kuiper_KB_AvgPool3D_avgpool3d_axis_fw_rm_f32(k, s, p, d, bc, l, l_out,
                                                 input, output);
    smul_fw_f32(Kuiper_KB_AvgPool3D_avgpool_recip_f32(k), bc * l_out, output);
    return (KRML_CLITERAL(Kuiper_KB_AvgPool3D_avgpool3d_axis_alloc_result){
        .l_out = l_out, .output = output});
}

__global__
/**
  hoisted when extracting avgpool3d_full_alloc_f32
*/
static void
__hoisted_avgpool3d_full_alloc_f32_0(uint32_t kh, uint32_t sh, uint32_t ph,
                                     uint32_t dh, uint32_t h, uint32_t wo,
                                     float *mid_h_in, uint32_t ho,
                                     uint32_t rows_h, float *mid_h)
{
    if (1024U * blockIdx.x + threadIdx.x < rows_h * ho) {
        uint32_t r_sz = (1024U * blockIdx.x + threadIdx.x) / ho;
        uint32_t j_sz = (1024U * blockIdx.x + threadIdx.x) % ho;
        float acc = 0.0f;
        uint32_t di_ref = 0U;
        float buf = 0.0f;
        KRML_HOST_IGNORE(&buf);
        for (; di_ref < kh; di_ref++) {
            uint32_t pos = j_sz * sh + di_ref * dh;
            bool in_bounds = pos >= ph && pos - ph < h;
            uint32_t dpos_ref = 0U;
            if (in_bounds)
                dpos_ref = pos - ph;
            float raw =
                mid_h_in[r_sz / wo * h * wo + dpos_ref * wo + r_sz % wo];
            acc += in_bounds ? raw : 0.0f;
        }
        mid_h[r_sz / wo * ho * wo + j_sz * wo + r_sz % wo] = acc;
    }
}

__global__
/**
  hoisted when extracting avgpool3d_full_alloc_f32
*/
static void
__hoisted_avgpool3d_full_alloc_f32_1(uint32_t kd, uint32_t sd, uint32_t pd,
                                     uint32_t dd, uint32_t depth, uint32_t wo,
                                     uint32_t ho, float *mid_d_in, uint32_t do_,
                                     uint32_t rows_d, float *out)
{
    if (1024U * blockIdx.x + threadIdx.x < rows_d * do_) {
        uint32_t r_sz = (1024U * blockIdx.x + threadIdx.x) / do_;
        uint32_t j_sz = (1024U * blockIdx.x + threadIdx.x) % do_;
        float acc = 0.0f;
        uint32_t di_ref = 0U;
        float buf = 0.0f;
        KRML_HOST_IGNORE(&buf);
        for (; di_ref < kd; di_ref++) {
            uint32_t pos = j_sz * sd + di_ref * dd;
            bool in_bounds = pos >= pd && pos - pd < depth;
            uint32_t dpos_ref = 0U;
            if (in_bounds)
                dpos_ref = pos - pd;
            float raw = mid_d_in[r_sz / (ho * wo) * depth * ho * wo +
                                 dpos_ref * ho * wo + r_sz % (ho * wo)];
            acc += in_bounds ? raw : 0.0f;
        }
        out[r_sz / (ho * wo) * do_ * ho * wo + j_sz * ho * wo +
            r_sz % (ho * wo)] = acc;
    }
}

Kuiper_KB_AvgPool3D_avgpool3d_full_result
Kuiper_KB_AvgPool3D_avgpool3d_full_alloc_f32(
    uint32_t kd, uint32_t kh, uint32_t kw, uint32_t sd, uint32_t sh,
    uint32_t sw, uint32_t pd, uint32_t ph, uint32_t pw, uint32_t dd,
    uint32_t dh, uint32_t dw, uint32_t bc, uint32_t depth, uint32_t h,
    uint32_t w, float *input)
{
    uint32_t wo = Kuiper_KB_AvgPool3D_pool_out_len_1d_sz(w, kw, sw, pw, dw);
    uint32_t rows_w = bc * depth * h;
    float *mid_w = (float *) KPR_GPU_ALLOC(sizeof(float), rows_w * wo);
    Kuiper_KB_AvgPool3D_avgpool3d_axis_fw_rm_f32(kw, sw, pw, dw, rows_w, w, wo,
                                                 input, mid_w);
    uint32_t n0 = rows_w * wo;
    smul_fw_f32(Kuiper_KB_AvgPool3D_avgpool_recip_f32(kw), n0, mid_w);
    float *mid_h_in = mid_w;
    uint32_t ho = Kuiper_KB_AvgPool3D_pool_out_len_1d_sz(h, kh, sh, ph, dh);
    uint32_t rows_h = bc * depth * wo;
    float *mid_h = (float *) KPR_GPU_ALLOC(sizeof(float), rows_h * ho);
    if (ho != 0U) {
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_avgpool3d_full_alloc_f32_0,
                  rows_h * ho / 1024U + (uint32_t) (rows_h * ho % 1024U != 0U),
                  1024U, 0U, s, kh, sh, ph, dh, h, wo, mid_h_in, ho, rows_h,
                  mid_h);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
    }
    MUST(cudaFree(mid_h_in));
    uint32_t n1 = rows_h * ho;
    smul_fw_f32(Kuiper_KB_AvgPool3D_avgpool_recip_f32(kh), n1, mid_h);
    float *mid_d_in = mid_h;
    uint32_t do_ =
        Kuiper_KB_AvgPool3D_pool_out_len_1d_sz(depth, kd, sd, pd, dd);
    uint32_t rows_d = bc * ho * wo;
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), rows_d * do_);
    if (do_ != 0U) {
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_avgpool3d_full_alloc_f32_1,
                  rows_d * do_ / 1024U +
                      (uint32_t) (rows_d * do_ % 1024U != 0U),
                  1024U, 0U, s, kd, sd, pd, dd, depth, wo, ho, mid_d_in, do_,
                  rows_d, out);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
    }
    MUST(cudaFree(mid_d_in));
    uint32_t n = rows_d * do_;
    smul_fw_f32(Kuiper_KB_AvgPool3D_avgpool_recip_f32(kd), n, out);
    return (KRML_CLITERAL(Kuiper_KB_AvgPool3D_avgpool3d_full_result){
        .w_out = wo, .h_out = ho, .d_out = do_, .full_output = out});
}

Kuiper_KB_AvgPool3D_avgpool3d_full_result
Kuiper_KB_AvgPool3D_avgpool3d_raw_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                            uint32_t b, uint32_t c,
                                            uint32_t depth, uint32_t h,
                                            uint32_t w, float *input)
{
    return Kuiper_KB_AvgPool3D_avgpool3d_full_alloc_f32(
        k, k, k, s, s, s, p, p, p, 1U, 1U, 1U, b * c, depth, h, w, input);
}
