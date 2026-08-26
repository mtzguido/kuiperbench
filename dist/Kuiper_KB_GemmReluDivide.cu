
#include "Kuiper_KB_GemmReluDivide.h"

__global__
/**
  hoisted when extracting gemm_relu_divide_f32
*/
static void
__hoisted_gemm_relu_divide_f32_0(uint32_t batch, uint32_t input, uint32_t out,
                                 float *x, float *wt, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < batch * out) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / out;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % out;
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < input; k++) {
            uint32_t vk = k;
            sum += x[trow * input + vk] * wt[vk * out + tcol];
        }
        gC[trow * out + tcol] = sum;
    }
}

__global__
/**
  hoisted when extracting gemm_relu_divide_f32
*/
static void
__hoisted_gemm_relu_divide_f32_1(uint32_t batch, uint32_t out, float *bias,
                                 float *y, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < batch * out) {
        uint32_t j = (1024U * blockIdx.x + threadIdx.x) % out;
        y[1024U * blockIdx.x + threadIdx.x] =
            gC[(1024U * blockIdx.x + threadIdx.x) / out * out + j] + bias[j];
    }
}

__global__
/**
  hoisted when extracting gemm_relu_divide_f32
*/
static void
__hoisted_gemm_relu_divide_f32_2(uint32_t batch, uint32_t out, float *y)
{
    if (1024U * blockIdx.x + threadIdx.x < batch * out)
        y[1024U * blockIdx.x + threadIdx.x] =
            fmaxf(y[1024U * blockIdx.x + threadIdx.x], 0.0f);
}

__global__
/**
  hoisted when extracting gemm_relu_divide_f32
*/
static void
__hoisted_gemm_relu_divide_f32_3(uint32_t batch, uint32_t out, float divisor,
                                 float *y)
{
    if (1024U * blockIdx.x + threadIdx.x < batch * out)
        y[1024U * blockIdx.x + threadIdx.x] /= divisor;
}

void Kuiper_KB_GemmReluDivide_gemm_relu_divide_f32(uint32_t batch,
                                                   uint32_t input, uint32_t out,
                                                   float divisor, float *x,
                                                   float *wt, float *bias,
                                                   float *y)
{
    float *gC = (float *) KPR_GPU_ALLOC(sizeof(float), batch * out);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_relu_divide_f32_0,
              batch * out / 1024U + (uint32_t) (batch * out % 1024U != 0U),
              1024U, 0U, s, batch, input, out, x, wt, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_relu_divide_f32_1,
              batch * out / 1024U + (uint32_t) (batch * out % 1024U != 0U),
              1024U, 0U, s0, batch, out, bias, y, gC);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_relu_divide_f32_2,
              batch * out / 1024U + (uint32_t) (batch * out % 1024U != 0U),
              1024U, 0U, s1, batch, out, y);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_relu_divide_f32_3,
              batch * out / 1024U + (uint32_t) (batch * out % 1024U != 0U),
              1024U, 0U, s2, batch, out, divisor, y);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(gC));
}
