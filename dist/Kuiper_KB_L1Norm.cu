
#include "Kuiper_KB_L1Norm.h"

__global__
/**
  hoisted when extracting l1norm_fw
*/
static void
__hoisted_l1norm_fw_0(uint32_t b, uint32_t d, float *x, float *sum_abs)
{
    if (1024U * blockIdx.x + threadIdx.x < b) {
        uint32_t ci_ref = 0U;
        float acc_ref = 0.0f;
        for (; ci_ref < d; ci_ref++) {
            float v = x[(1024U * blockIdx.x + threadIdx.x) * d + ci_ref];
            float acc_v = acc_ref;
            acc_ref = acc_v + fmaxf(v, 0.0f - v);
        }
        sum_abs[1024U * blockIdx.x + threadIdx.x] = acc_ref;
    }
}

__global__
/**
  hoisted when extracting l1norm_fw
*/
static void
__hoisted_l1norm_fw_1(uint32_t b, float dim_f, float *sum_abs)
{
    if (1024U * blockIdx.x + threadIdx.x < b)
        sum_abs[1024U * blockIdx.x + threadIdx.x] =
            dim_f / sum_abs[1024U * blockIdx.x + threadIdx.x];
}

__global__
/**
  hoisted when extracting l1norm_fw
*/
static void
__hoisted_l1norm_fw_2(uint32_t b, uint32_t d, float *x, float *sum_abs)
{
    if (1024U * blockIdx.x + threadIdx.x < b * d) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / d;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % d;
        x[row * d + col] *= sum_abs[row];
    }
}

void Kuiper_KB_L1Norm_l1norm_fw(uint32_t b, uint32_t d, float *x)
{
    float dim_f = (float) (int64_t) (uint64_t) d;
    float *sum_abs = (float *) KPR_GPU_ALLOC(sizeof(float), b);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_l1norm_fw_0, b / 1024U + (uint32_t) (b % 1024U != 0U),
              1024U, 0U, s, b, d, x, sum_abs);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_l1norm_fw_1, b / 1024U + (uint32_t) (b % 1024U != 0U),
              1024U, 0U, s0, b, dim_f, sum_abs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_l1norm_fw_2,
              b * d / 1024U + (uint32_t) (b * d % 1024U != 0U), 1024U, 0U, s1,
              b, d, x, sum_abs);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaFree(sum_abs));
}

void (*Kuiper_KB_L1Norm_l1norm_fw_f32)(uint32_t x0, uint32_t x1,
                                       float *x2) = Kuiper_KB_L1Norm_l1norm_fw;

float *Kuiper_KB_L1Norm_l1norm_alloc_f32(uint32_t b, uint32_t d, float *x)
{
    uint32_t n = b * d;
    float *dst = (float *) KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(dst, x, (uint32_t) sizeof(float) * n,
                    cudaMemcpyDeviceToDevice));
    float *out = dst;
    Kuiper_KB_L1Norm_l1norm_fw_f32(b, d, out);
    return out;
}
