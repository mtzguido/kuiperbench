
#include "Kuiper_KB_Elu.h"

__global__
/**
  hoisted when extracting elu_fw_f32
*/
static void
__hoisted_elu_fw_f32_0(float alpha, uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float x = a[1024U * blockIdx.x + threadIdx.x];
        float ite;
        if ((float) 0LL < x)
            ite = x;
        else
            ite = alpha * (expf(x) - (float) 1LL);
        a[1024U * blockIdx.x + threadIdx.x] = ite;
    }
}

void Kuiper_KB_Elu_elu_fw_f32(float alpha, uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_elu_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              alpha, lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting elu_fw_f64
*/
static void
__hoisted_elu_fw_f64_0(double alpha, uint32_t lena, double *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double x = a[1024U * blockIdx.x + threadIdx.x];
        double ite;
        if ((double) 0LL < x)
            ite = x;
        else
            ite = alpha * (exp(x) - (double) 1LL);
        a[1024U * blockIdx.x + threadIdx.x] = ite;
    }
}

void Kuiper_KB_Elu_elu_fw_f64(double alpha, uint32_t lena, double *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_elu_fw_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              alpha, lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting elu_alloc_f32
*/
static void
__hoisted_elu_alloc_f32_0(float alpha, uint32_t lena, float *input,
                          float *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float y = input[1024U * blockIdx.x + threadIdx.x];
        float ite;
        if ((float) 0LL < y)
            ite = y;
        else
            ite = alpha * (expf(y) - (float) 1LL);
        output[1024U * blockIdx.x + threadIdx.x] = ite;
    }
}

float *Kuiper_KB_Elu_elu_alloc_f32(float alpha, uint32_t lena, float *input)
{
    float *_return;
    bool _return1 = false;
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_elu_alloc_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              alpha, lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}

__global__
/**
  hoisted when extracting elu_alloc_f64
*/
static void
__hoisted_elu_alloc_f64_0(double alpha, uint32_t lena, double *input,
                          double *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double y = input[1024U * blockIdx.x + threadIdx.x];
        double ite;
        if ((double) 0LL < y)
            ite = y;
        else
            ite = alpha * (exp(y) - (double) 1LL);
        output[1024U * blockIdx.x + threadIdx.x] = ite;
    }
}

double *Kuiper_KB_Elu_elu_alloc_f64(double alpha, uint32_t lena, double *input)
{
    double *_return;
    bool _return1 = false;
    double *output = (double *) KPR_GPU_ALLOC(sizeof(double), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_elu_alloc_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              alpha, lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}

__global__
/**
  hoisted when extracting elu_alloc_f64_f32
*/
static void
__hoisted_elu_alloc_f64_f32_0(double alpha, uint32_t lena, float *input,
                              float *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float y = input[1024U * blockIdx.x + threadIdx.x];
        float ite;
        if ((float) 0LL < y)
            ite = y;
        else
            ite = (float) alpha * (expf(y) - (float) 1LL);
        output[1024U * blockIdx.x + threadIdx.x] = ite;
    }
}

float *Kuiper_KB_Elu_elu_alloc_f64_f32(double alpha, uint32_t lena,
                                       float *input)
{
    float *_return;
    bool _return1 = false;
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_elu_alloc_f64_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              alpha, lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}
