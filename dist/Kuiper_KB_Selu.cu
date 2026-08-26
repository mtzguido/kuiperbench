
#include "Kuiper_KB_Selu.h"

__global__
/**
  hoisted when extracting selu_fw_f32
*/
static void
__hoisted_selu_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        float ite;
        if (0.0f < x)
            ite = x;
        else
            ite = 1.6732632423543772848 * (expf(x) - 1.0f);
        a[1024U * blockIdx.x + threadIdx.x] = 1.0507009873554804934 * ite;
    }
}

void Kuiper_KB_Selu_selu_fw_f32(uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_selu_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting selu_fw_f64
*/
static void
__hoisted_selu_fw_f64_0(uint32_t lena, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        double ite;
        if (0.0 < x)
            ite = x;
        else
            ite = 1.6732632423543772848 * (exp(x) - 1.0);
        a[1024U * blockIdx.x + threadIdx.x] = 1.0507009873554804934 * ite;
    }
}

void Kuiper_KB_Selu_selu_fw_f64(uint32_t lena, double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_selu_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}
