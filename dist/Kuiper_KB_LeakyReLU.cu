
#include "Kuiper_KB_LeakyReLU.h"

__global__
/**
  hoisted when extracting leaky_relu_fw_f32
*/
static void
__hoisted_leaky_relu_fw_f32_0(float slope, uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = 0.0f < x ? x : x * slope;
    }
}

void Kuiper_KB_LeakyReLU_leaky_relu_fw_f32(float slope, uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_leaky_relu_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              slope, lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting leaky_relu_fw_f64
*/
static void
__hoisted_leaky_relu_fw_f64_0(double slope, uint32_t lena, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] = 0.0 < x ? x : x * slope;
    }
}

void Kuiper_KB_LeakyReLU_leaky_relu_fw_f64(double slope, uint32_t lena,
                                           double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_leaky_relu_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              slope, lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}
