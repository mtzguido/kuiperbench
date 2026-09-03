
#include "Kuiper_KB_TransposedGEMM.h"

__global__
/**
  hoisted when extracting matmul_f32_atb
*/
static void
__hoisted_matmul_f32_atb_0(uint32_t m, uint32_t n, uint32_t k, float *gA,
                           float *gB, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            float old_acc = acc;
            float yc = gA[__anf0 * m + trow] * gB[__anf0 * n + tcol] - c;
            float t = old_acc + yc;
            c = t - old_acc - yc;
            acc = t;
        }
        gC[trow * n + tcol] = acc;
    }
}

void Kuiper_KB_TransposedGEMM_matmul_f32_atb(uint32_t m, uint32_t n, uint32_t k,
                                             float *gA, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_matmul_f32_atb_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s, m,
              n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting matmul_f32_abt
*/
static void
__hoisted_matmul_f32_abt_0(uint32_t m, uint32_t n, uint32_t k, float *gA,
                           float *gB, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            float old_acc = acc;
            float yc = gA[trow * k + __anf0] * gB[tcol * k + __anf0] - c;
            float t = old_acc + yc;
            c = t - old_acc - yc;
            acc = t;
        }
        gC[trow * n + tcol] = acc;
    }
}

void Kuiper_KB_TransposedGEMM_matmul_f32_abt(uint32_t m, uint32_t n, uint32_t k,
                                             float *gA, float *gB, float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_matmul_f32_abt_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s, m,
              n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting matmul_f32_atbt
*/
static void
__hoisted_matmul_f32_atbt_0(uint32_t m, uint32_t n, uint32_t k, float *gA,
                            float *gB, float *gC)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t trow = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t tcol = (1024U * blockIdx.x + threadIdx.x) % n;
        uint32_t k1 = 0U;
        float acc = 0.0f;
        float c = 0.0f;
        for (; k1 < k; k1++) {
            uint32_t __anf0 = k1;
            float old_acc = acc;
            float yc = gA[__anf0 * m + trow] * gB[tcol * k + __anf0] - c;
            float t = old_acc + yc;
            c = t - old_acc - yc;
            acc = t;
        }
        gC[trow * n + tcol] = acc;
    }
}

void Kuiper_KB_TransposedGEMM_matmul_f32_atbt(uint32_t m, uint32_t n,
                                              uint32_t k, float *gA, float *gB,
                                              float *gC)
{
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_matmul_f32_atbt_0,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s, m,
              n, k, gA, gB, gC);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

float *Kuiper_KB_TransposedGEMM_matmul_f32_atb_alloc(uint32_t m, uint32_t n,
                                                     uint32_t k, float *gA,
                                                     float *gB)
{
    float *gC = (float *) KPR_GPU_ALLOC(sizeof(float), m * n);
    Kuiper_KB_TransposedGEMM_matmul_f32_atb(m, n, k, gA, gB, gC);
    return gC;
}

float *Kuiper_KB_TransposedGEMM_matmul_f32_abt_alloc(uint32_t m, uint32_t n,
                                                     uint32_t k, float *gA,
                                                     float *gB)
{
    float *gC = (float *) KPR_GPU_ALLOC(sizeof(float), m * n);
    Kuiper_KB_TransposedGEMM_matmul_f32_abt(m, n, k, gA, gB, gC);
    return gC;
}

float *Kuiper_KB_TransposedGEMM_matmul_f32_atbt_alloc(uint32_t m, uint32_t n,
                                                      uint32_t k, float *gA,
                                                      float *gB)
{
    float *gC = (float *) KPR_GPU_ALLOC(sizeof(float), m * n);
    Kuiper_KB_TransposedGEMM_matmul_f32_atbt(m, n, k, gA, gB, gC);
    return gC;
}
