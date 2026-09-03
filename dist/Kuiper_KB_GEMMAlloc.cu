
#include "Kuiper_KB_GEMMAlloc.h"

__global__
/**
  hoisted when extracting gemm_naive3_alloc_f32
*/
static void
__hoisted_gemm_naive3_alloc_f32_0(uint32_t m, uint32_t n, uint32_t k, float *a,
                                  float *b, float *c)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c1 = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            float old_acc = acc;
            float yc = a[trow * k + __anf0] * b[__anf0 * n + tcol] - c1;
            float t = old_acc + yc;
            c1 = t - old_acc - yc;
            acc = t;
        }
        c[trow * n + tcol] = acc;
    }
}

float *Kuiper_KB_GEMMAlloc_gemm_naive3_alloc_f32(uint32_t m, uint32_t n,
                                                 uint32_t k, float *a, float *b)
{
    KPR_GUARD(k <= 4294967295U / m);
    KPR_GUARD(n <= 4294967295U / k);
    KPR_GUARD(n <= 4294967295U / m);
    KPR_GUARD(m * n <= 2147483648U);
    float *c = (float *) KPR_GPU_ALLOC(sizeof(float), m * n);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_naive3_alloc_f32_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s, m,
              n, k, a, b, c);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return c;
}

__global__
/**
  hoisted when extracting gemm_naive1_alloc_f32
*/
static void
__hoisted_gemm_naive1_alloc_f32_0(uint32_t n, uint32_t k, float *a, float *b,
                                  float *c)
{
    uint32_t trow = blockIdx.x / n;
    uint32_t tcol = blockIdx.x % n;
    uint32_t k1 = 0U;
    float sum = 0.0f;
    for (; k1 < k; k1++) {
        uint32_t vk = k1;
        sum += a[trow * k + vk] * b[vk * n + tcol];
    }
    c[trow * n + tcol] = sum;
}

float *Kuiper_KB_GEMMAlloc_gemm_naive1_alloc_f32(uint32_t m, uint32_t n,
                                                 uint32_t k, float *a, float *b)
{
    KPR_GUARD(k <= 4294967295U / m);
    KPR_GUARD(n <= 4294967295U / k);
    KPR_GUARD(n <= 4294967295U / m);
    KPR_GUARD(m * n <= 2097152U);
    float *c = (float *) KPR_GPU_ALLOC(sizeof(float), m * n);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_gemm_naive1_alloc_f32_0, m * n, 1U, 0U, s, n, k, a, b,
              c);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return c;
}
