
#include "Kuiper_KB_MaskedCumSum.h"

typedef struct __uint32_t__uint32_t_______s {
    uint32_t fst;
    uint32_t snd;
} __uint32_t__uint32_t______;

__global__
/**
  hoisted when extracting masked_cumsum_fw_f32
*/
static void
__hoisted_masked_cumsum_fw_f32_0(uint32_t d, uint32_t n1, float *input_f,
                                 uint8_t *mask_f, float *scratch_f)
{
    if (1024U * blockIdx.x + threadIdx.x < n1) {
        __uint32_t__uint32_t______ scrut0 = {
            .fst = (1024U * blockIdx.x + threadIdx.x) / d,
            .snd = (1024U * blockIdx.x + threadIdx.x) % d};
        __uint32_t__uint32_t______ scrut1 = {
            .fst = (1024U * blockIdx.x + threadIdx.x) / d,
            .snd = (1024U * blockIdx.x + threadIdx.x) % d};
        float x1 = input_f[scrut1.fst * d + scrut1.snd];
        __uint32_t__uint32_t______ scrut = {
            .fst = (1024U * blockIdx.x + threadIdx.x) / d,
            .snd = (1024U * blockIdx.x + threadIdx.x) % d};
        float ite;
        if (mask_f[scrut.fst * d + scrut.snd] == 0U)
            ite = 0.0f;
        else
            ite = x1;
        scratch_f[scrut0.fst * d + scrut0.snd] = ite;
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

static float *tensor_apply_bij_ro_located__float(float *a) { return a; }

static uint8_t *tensor_apply_bij_ro_located__uint8_t(uint8_t *a) { return a; }

static float *tensor_apply_bij_st_located__float(float *a) { return a; }

float *Kuiper_KB_MaskedCumSum_masked_cumsum_fw_f32(uint32_t b, uint32_t d,
                                                   float *input, uint8_t *mask)
{
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), b * d);
    uint32_t n1 = b * d;
    float *scratch = (float *) KPR_GPU_ALLOC(sizeof(float), n1);
    float *input_f = tensor_apply_bij_ro_located__float(input);
    uint8_t *mask_f = tensor_apply_bij_ro_located__uint8_t(mask);
    float *scratch_f = tensor_apply_bij_st_located__float(scratch);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_masked_cumsum_fw_f32_0,
              n1 / 1024U + (uint32_t) (n1 % 1024U != 0U), 1024U, 0U, s, d, n1,
              input_f, mask_f, scratch_f);
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
