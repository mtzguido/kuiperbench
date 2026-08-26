
#include "Kuiper_KB_NewGelu.h"

__global__
/**
  hoisted when extracting newgelu_fw_f32
*/
static void
__hoisted_newgelu_fw_f32_0(float half, float c, float k, uint32_t lena,
                           float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] =
            half * x * (1.0f + tanhf(c * (x + k * (x * x * x))));
    }
}

void Kuiper_KB_NewGelu_newgelu_fw_f32(float half, float c, float k,
                                      uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_newgelu_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              half, c, k, lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting newgelu_fw_f64
*/
static void
__hoisted_newgelu_fw_f64_0(double half, double c, double k, uint32_t lena,
                           double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] =
            half * x * (1.0 + tanh(c * (x + k * (x * x * x))));
    }
}

void Kuiper_KB_NewGelu_newgelu_fw_f64(double half, double c, double k,
                                      uint32_t lena, double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_newgelu_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              half, c, k, lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}
