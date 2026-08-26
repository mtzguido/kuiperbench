
#include "Kuiper_KB_CumSum.h"

__global__
/**
  hoisted when extracting cumsum_fw_f32
*/
static void
__hoisted_cumsum_fw_f32_0(uint32_t d, float *input, float *output)
{
    float acc = 0.0f;
    uint32_t di_ref = 0U;
    for (; di_ref < d; di_ref++) {
        uint32_t di_old_sz = di_ref;
        acc += input[blockIdx.x * d + di_old_sz];
        output[blockIdx.x * d + di_old_sz] = acc;
    }
}

void Kuiper_KB_CumSum_cumsum_fw_f32(uint32_t b, uint32_t d, float *input,
                                    float *output)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_cumsum_fw_f32_0, b, 1U, 0U, s, d, input, output);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
