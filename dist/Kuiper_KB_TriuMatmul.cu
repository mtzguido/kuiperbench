
#include "Kuiper_KB_TriuMatmul.h"

__global__
/**
  hoisted when extracting triu_matmul_f32
*/
static void
__hoisted_triu_matmul_f32_0(uint32_t n, float *gA, float *gB, float *y)
{
    if (1024U * blockIdx.x + threadIdx.x < n * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k < n; k++) {
            uint32_t __anf0 = k;
            float old_acc = acc;
            float yc = gA[trow * n + __anf0] * gB[__anf0 * n + tcol] - c;
            float t = old_acc + yc;
            c = t - old_acc - yc;
            acc = t;
        }
        y[trow * n + tcol] = acc;
    }
}

__global__
/**
  hoisted when extracting triu_matmul_f32
*/
static void
__hoisted_triu_matmul_f32_1(uint32_t n, float *y)
{
    if (1024U * blockIdx.x + threadIdx.x < n * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        float x = y[row * n + col];
        if (row <= col)
            y[row * n + col] = x;
        else
            y[row * n + col] = 0.0f;
    }
}

void Kuiper_KB_TriuMatmul_triu_matmul_f32(uint32_t n, float *gA, float *gB,
                                          float *y)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_triu_matmul_f32_0,
              n * n / 1024U + (uint32_t) (n * n % 1024U != 0U), 1024U, 0U, s, n,
              gA, gB, y);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_triu_matmul_f32_1,
              n * n / 1024U + (uint32_t) (n * n % 1024U != 0U), 1024U, 0U, s0,
              n, y);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
}
