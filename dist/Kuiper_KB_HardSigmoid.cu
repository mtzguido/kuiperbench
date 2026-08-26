
#include "Kuiper_KB_HardSigmoid.h"

__global__
/**
  hoisted when extracting hsig_fw_f32
*/
static void
__hoisted_hsig_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] =
            (float) 3LL <= x ? 1.0f
            : x <= 0.0f - (float) 3LL
                ? 0.0f
                : x * (1.0f / (float) 6LL) + 1.0f / (float) 2LL;
    }
}

void Kuiper_KB_HardSigmoid_hsig_fw_f32(uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_hsig_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting hsig_fw_f64
*/
static void
__hoisted_hsig_fw_f64_0(uint32_t lena, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] =
            (double) 3LL <= x ? 1.0
            : x <= 0.0 - (double) 3LL
                ? 0.0
                : x * (1.0 / (double) 6LL) + 1.0 / (double) 2LL;
    }
}

void Kuiper_KB_HardSigmoid_hsig_fw_f64(uint32_t lena, double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_hsig_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}
