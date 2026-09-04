
#include "Kuiper_KB_CumSumReverse.h"

__global__
/**
  hoisted when extracting cumsum_reverse_fw_f32
*/
static void
__hoisted_cumsum_reverse_fw_f32_0(uint32_t d, float *input_r, float *output_r)
{
    float acc = 0.0f;
    uint32_t di_ref = 0U;
    for (; di_ref < d; di_ref++) {
        uint32_t di_old_sz = di_ref;
        acc += input_r[blockIdx.x * d + (d - 1U - di_old_sz)];
        output_r[blockIdx.x * d + (d - 1U - di_old_sz)] = acc;
    }
}

static float *tensor_apply_bij_ro_located__float(float *a) { return a; }

static float *tensor_apply_bij_st_located__float(float *a) { return a; }

float *Kuiper_KB_CumSumReverse_cumsum_reverse_fw_f32(uint32_t b, uint32_t d,
                                                     float *input)
{
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), b * d);
    float *input_r = tensor_apply_bij_ro_located__float(input);
    float *output_r = tensor_apply_bij_st_located__float(output);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_cumsum_reverse_fw_f32_0, b, 1U, 0U, s, d, input_r,
              output_r);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return output;
}
