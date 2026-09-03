
#include "Kuiper_KB_AvgPool1D.h"

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

uint32_t Kuiper_KB_AvgPool1D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
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

float Kuiper_KB_AvgPool1D_avgpool_recip_f32(uint32_t k)
{
    return 1.0f / (float) (int64_t) (uint64_t) k;
}

__global__
/**
  hoisted when extracting avgpool1d_fw_rm_f32
*/
static void
__hoisted_avgpool1d_fw_rm_f32_0(uint32_t k, uint32_t s, uint32_t p, uint32_t d,
                                uint32_t bc, uint32_t l, uint32_t l_out,
                                float *input, float *output)
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

void Kuiper_KB_AvgPool1D_avgpool1d_fw_rm_f32(uint32_t k, uint32_t s, uint32_t p,
                                             uint32_t d, uint32_t bc,
                                             uint32_t l, uint32_t l_out,
                                             float *input, float *output)
{
    if (l_out != 0U) {
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_avgpool1d_fw_rm_f32_0,
                  bc * l_out / 1024U + (uint32_t) (bc * l_out % 1024U != 0U),
                  1024U, 0U, s1, k, s, p, d, bc, l, l_out, input, output);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
    }
}

Prims_dtuple2__uint32_t__float_
Kuiper_KB_AvgPool1D_avgpool1d_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                        uint32_t d, uint32_t bc, uint32_t l,
                                        float *input)
{
    uint32_t l_out = Kuiper_KB_AvgPool1D_pool_out_len_1d_sz(l, k, s, p, d);
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), bc * l_out);
    Kuiper_KB_AvgPool1D_avgpool1d_fw_rm_f32(k, s, p, d, bc, l, l_out, input,
                                            output);
    smul_fw_f32(Kuiper_KB_AvgPool1D_avgpool_recip_f32(k), bc * l_out, output);
    return (KRML_CLITERAL(Prims_dtuple2__uint32_t__float_){.fst = l_out,
                                                           .snd = output});
}

Prims_dtuple2__uint32_t__float_
Kuiper_KB_AvgPool1D_avgpool1d_raw_alloc_f32(uint32_t k, uint32_t s, uint32_t p,
                                            uint32_t b, uint32_t c, uint32_t l,
                                            float *input)
{
    return Kuiper_KB_AvgPool1D_avgpool1d_alloc_f32(k, s, p, 1U, b * c, l,
                                                   input);
}
