
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

__global__
/**
  hoisted when extracting newgelu_alloc_f32
*/
static void
__hoisted_newgelu_alloc_f32_0(uint32_t lena, float *input, float *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        float y = input[1024U * blockIdx.x + threadIdx.x];
        output[1024U * blockIdx.x + threadIdx.x] =
            0.5 * y *
            (1.0f +
             tanhf(0.79788456080286535588 * (y + 0.044715 * (y * y * y))));
    }
}

float *Kuiper_KB_NewGelu_newgelu_alloc_f32(uint32_t lena, float *input)
{
    float *_return;
    bool _return1 = false;
    float *output = (float *) KPR_GPU_ALLOC(sizeof(float), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_newgelu_alloc_f32_0,
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
  hoisted when extracting newgelu_alloc_f64
*/
static void
__hoisted_newgelu_alloc_f64_0(uint32_t lena, double *input, double *output)
{
    if (1024U * blockIdx.x + threadIdx.x < lena) {
        double y = input[1024U * blockIdx.x + threadIdx.x];
        output[1024U * blockIdx.x + threadIdx.x] =
            0.5 * y *
            (1.0 + tanh(0.79788456080286535588 * (y + 0.044715 * (y * y * y))));
    }
}

double *Kuiper_KB_NewGelu_newgelu_alloc_f64(uint32_t lena, double *input)
{
    double *_return;
    bool _return1 = false;
    double *output = (double *) KPR_GPU_ALLOC(sizeof(double), lena);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_newgelu_alloc_f64_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1,
              lena, input, output);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    _return = output;
    _return1 = true;
    return _return;
}
