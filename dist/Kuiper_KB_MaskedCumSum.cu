
#include "Kuiper_KB_MaskedCumSum.h"

__global__
/**
  hoisted when extracting masked_cumsum_fw_f32
*/
static void
__hoisted_masked_cumsum_fw_f32_0(float *input, uint8_t *mask, uint32_t n1,
                                 float *scratch)
{
    if (1024U * blockIdx.x + threadIdx.x < n1) {
        float x1 = input[1024U * blockIdx.x + threadIdx.x];
        scratch[1024U * blockIdx.x + threadIdx.x] =
            mask[1024U * blockIdx.x + threadIdx.x] == 0U ? 0.0f : x1;
    }
}

__global__
/**
  hoisted when extracting masked_cumsum_fw_f32
*/
static void
__hoisted_masked_cumsum_fw_f32_1(uint32_t d, float *output, float *scratch)
{
    float acc = 0.0f;
    uint32_t di_ref = 0U;
    for (; di_ref < d; di_ref++) {
        uint32_t di_old_sz = di_ref;
        acc += scratch[blockIdx.x * d + di_old_sz];
        output[blockIdx.x * d + di_old_sz] = acc;
    }
}

float *Kuiper_KB_MaskedCumSum_masked_cumsum_fw_f32(uint32_t b, uint32_t d,
                                                   float *input, uint8_t *mask)
{
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), b * d);
    uint32_t n1 = b * d;
    float *scratch = (float *) KPR_GPU_ALLOC(sizeof(float), n1);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_masked_cumsum_fw_f32_0,
              n1 / 1024U + (uint32_t) (n1 % 1024U != 0U), 1024U, 0U, s, input,
              mask, n1, scratch);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_masked_cumsum_fw_f32_1, b, 1U, 0U, s0, d, output,
              scratch);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    MUST(cudaFree(scratch));
    return output;
}
