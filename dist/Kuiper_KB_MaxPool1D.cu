
#include "Kuiper_KB_MaxPool1D.h"

uint32_t Kuiper_KB_MaxPool1D_pool_out_len_1d_sz(uint32_t l, uint32_t k,
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
  hoisted when extracting maxpool1d_fw_rm_f32
*/
static void
__hoisted_maxpool1d_fw_rm_f32_0(uint32_t k, uint32_t s, uint32_t p, uint32_t d,
                                uint32_t bc, uint32_t l, uint32_t l_out,
                                float *input, float *output)
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

void Kuiper_KB_MaxPool1D_maxpool1d_fw_rm_f32(uint32_t k, uint32_t s, uint32_t p,
                                             uint32_t d, uint32_t bc,
                                             uint32_t l, uint32_t l_out,
                                             float *input, float *output)
{
    if (l_out != 0U) {
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_maxpool1d_fw_rm_f32_0,
                  bc * l_out / 1024U + (uint32_t) (bc * l_out % 1024U != 0U),
                  1024U, 0U, s1, k, s, p, d, bc, l, l_out, input, output);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
    }
}

Kuiper_KB_MaxPool1D_maxpool1d_alloc_result
Kuiper_KB_MaxPool1D_maxpool1d_alloc_f32(uint32_t b, uint32_t c, uint32_t l,
                                        uint32_t k, uint32_t s, uint32_t p,
                                        uint32_t d, float *input)
{
    uint32_t l_out = Kuiper_KB_MaxPool1D_pool_out_len_1d_sz(l, k, s, p, d);
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), b * c * l_out);
    Kuiper_KB_MaxPool1D_maxpool1d_fw_rm_f32(k, s, p, d, b * c, l, l_out, input,
                                            output);
    return (KRML_CLITERAL(Kuiper_KB_MaxPool1D_maxpool1d_alloc_result){
        .l_out = l_out, .output = output});
}
