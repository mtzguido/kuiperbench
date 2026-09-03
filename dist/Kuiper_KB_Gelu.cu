
#include "Kuiper_KB_Gelu.h"

__global__
/**
  hoisted when extracting gelu_fw_f32
*/
static void
__hoisted_gelu_fw_f32_0(uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] =
            1.0f / (float) 2LL * x *
            (1.0f + erff(x * (1.0f / sqrtf((float) 2LL))));
    }
}

void Kuiper_KB_Gelu_gelu_fw_f32(uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gelu_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting gelu_fw_f64
*/
static void
__hoisted_gelu_fw_f64_0(uint32_t lena, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        a[1024U * blockIdx.x + threadIdx.x] =
            1.0 / (double) 2LL * x *
            (1.0 + erf(x * (1.0 / sqrt((double) 2LL))));
    }
}

void Kuiper_KB_Gelu_gelu_fw_f64(uint32_t lena, double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gelu_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting gelu_alloc_f32
*/
static void
__hoisted_gelu_alloc_f32_0(uint32_t lena, float *input, float *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float y = input[1024U * blockIdx.x + threadIdx.x];
        output[1024U * blockIdx.x + threadIdx.x] =
            1.0f / (float) 2LL * y *
            (1.0f + erff(y * (1.0f / sqrtf((float) 2LL))));
    }
}

float *Kuiper_KB_Gelu_gelu_alloc_f32(uint32_t lena, float *input)
{
    float *_return;
    bool _return1 = false;
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gelu_alloc_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}

__global__
/**
  hoisted when extracting gelu_alloc_f64
*/
static void
__hoisted_gelu_alloc_f64_0(uint32_t lena, double *input, double *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double y = input[1024U * blockIdx.x + threadIdx.x];
        output[1024U * blockIdx.x + threadIdx.x] =
            1.0 / (double) 2LL * y *
            (1.0 + erf(y * (1.0 / sqrt((double) 2LL))));
    }
}

double *Kuiper_KB_Gelu_gelu_alloc_f64(uint32_t lena, double *input)
{
    double *_return;
    bool _return1 = false;
    double *output = (double *) KPR_GPU_ALLOC(sizeof(double), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gelu_alloc_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}
