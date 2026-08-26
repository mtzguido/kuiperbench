
#include "Kuiper_KB_Sigmoid.h"

__global__
/**
  hoisted when extracting sigmoid_fw_f32
*/
static void
__hoisted_sigmoid_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            1.0f / (1.0f + expf(0.0f - a[1024U * blockIdx.x + threadIdx.x]));
}

void Kuiper_KB_Sigmoid_sigmoid_fw_f32(uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_sigmoid_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting sigmoid_fw_f64
*/
static void
__hoisted_sigmoid_fw_f64_0(uint32_t lena, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] =
            1.0 / (1.0 + exp(0.0 - a[1024U * blockIdx.x + threadIdx.x]));
}

void Kuiper_KB_Sigmoid_sigmoid_fw_f64(uint32_t lena, double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_sigmoid_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}
