
#include "Kuiper_KB_GemmDivSumScale.h"

__global__
/**
  hoisted when extracting gemm_div_sum_scale_f32
*/
static void
__hoisted_gemm_div_sum_scale_f32_0(uint32_t batch, uint32_t input,
                                   uint32_t hidden, float *x, float *wt,
                                   float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < batch * hidden) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / hidden;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % hidden;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k1 < input; k1++) {
            uint32_t __anf0 = k1;
            float old_acc = acc;
            float yc =
                x[trow * input + __anf0] * wt[__anf0 * hidden + tcol] - c;
            float t = old_acc + yc;
            c = t - old_acc - yc;
            acc = t;
        }
        gC[trow * hidden + tcol] = acc;
    }
}

__global__
/**
  hoisted when extracting gemm_div_sum_scale_f32
*/
static void
__hoisted_gemm_div_sum_scale_f32_1(uint32_t hidden, float *y, float *gC)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < hidden; idx += 1024U)
        acc += gC[blockIdx.x * hidden + idx];
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        y[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting gemm_div_sum_scale_f32
*/
static void
__hoisted_gemm_div_sum_scale_f32_2(uint32_t batch, float k, float *y)
{
    if (1024U * blockIdx.x + threadIdx.x < batch)
        y[1024U * blockIdx.x + threadIdx.x] *= k;
}

void Kuiper_KB_GemmDivSumScale_gemm_div_sum_scale_f32(uint32_t batch,
                                                      uint32_t input,
                                                      uint32_t hidden, float k,
                                                      float *x, float *wt,
                                                      float *y)
{
    float *gC = (float *) KPR_GPU_ALLOC(sizeof(float), batch * hidden);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_div_sum_scale_f32_0,
              batch * hidden / 1024U +
                  (uint32_t) (batch * hidden % 1024U != 0U),
              1024U, 0U, s, batch, input, hidden, x, wt, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_gemm_div_sum_scale_f32_1, batch, 1024U, 4096U, s0,
              hidden, y, gC);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_div_sum_scale_f32_2,
              batch / 1024U + (uint32_t) (batch % 1024U != 0U), 1024U, 0U, s1,
              batch, k, y);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaFree(gC));
}
