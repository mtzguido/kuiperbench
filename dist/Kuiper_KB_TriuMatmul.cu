
#include "Kuiper_KB_TriuMatmul.h"

__global__
/**
  hoisted when extracting triu_matmul_f32
*/
static void
__hoisted_triu_matmul_f32_0(uint32_t n, float *gA, float *gB, float *y)
{
    if (1024U * blockIdx.x + threadIdx.x < n * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        float acc = 0.0f;
        uint32_t k = row;
        for (; k <= col; k++) {
            uint32_t kk = k;
            acc += gA[row * n + kk] * gB[kk * n + col];
        }
        y[row * n + col] = acc;
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
}

float *Kuiper_KB_TriuMatmul_triu_matmul_alloc_f32(uint32_t n, float *gA,
                                                  float *gB)
{
    float *y = (float *) KPR_GPU_ALLOC(sizeof(float), n * n);
    Kuiper_KB_TriuMatmul_triu_matmul_f32(n, gA, gB, y);
    return y;
}
