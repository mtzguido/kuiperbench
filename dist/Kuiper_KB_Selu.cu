
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

__global__
/**
  hoisted when extracting selu_alloc_f32
*/
static void
__hoisted_selu_alloc_f32_0(uint32_t lena, float *input, float *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float y = input[1024U * blockIdx.x + threadIdx.x];
        float ite;
        if (0.0f < y)
            ite = y;
        else
            ite = 1.6732632423543772848 * (expf(y) - 1.0f);
        output[1024U * blockIdx.x + threadIdx.x] = 1.0507009873554804934 * ite;
    }
}

float *Kuiper_KB_Selu_selu_alloc_f32(uint32_t lena, float *input)
{
    float *_return;
    bool _return1 = false;
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_selu_alloc_f32_0,
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
  hoisted when extracting selu_alloc_f64
*/
static void
__hoisted_selu_alloc_f64_0(uint32_t lena, double *input, double *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double y = input[1024U * blockIdx.x + threadIdx.x];
        double ite;
        if (0.0 < y)
            ite = y;
        else
            ite = 1.6732632423543772848 * (exp(y) - 1.0);
        output[1024U * blockIdx.x + threadIdx.x] = 1.0507009873554804934 * ite;
    }
}

double *Kuiper_KB_Selu_selu_alloc_f64(uint32_t lena, double *input)
{
    double *_return;
    bool _return1 = false;
    double *output = (double *) KPR_GPU_ALLOC(sizeof(double), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_selu_alloc_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}
