
#include "Kuiper_KB_CumSumExclusive.h"

__global__
/**
  hoisted when extracting cumsum_exclusive_fw_f32
*/
static void
__hoisted_cumsum_exclusive_fw_f32_0(uint32_t d, float *input, float *output)
{
    float acc = 0.0f;
    uint32_t di_ref = 0U;
    for (; di_ref < d; di_ref++) {
        uint32_t di_old_sz = di_ref;
        float v = input[blockIdx.x * d + di_old_sz];
        float acc_old = acc;
        output[blockIdx.x * d + di_old_sz] = acc_old;
        acc = acc_old + v;
    }
}

float *Kuiper_KB_CumSumExclusive_cumsum_exclusive_fw_f32(uint32_t b, uint32_t d,
                                                         float *input)
{
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), b * d);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_cumsum_exclusive_fw_f32_0, b, 1U, 0U, s, d, input,
              output);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return output;
}
