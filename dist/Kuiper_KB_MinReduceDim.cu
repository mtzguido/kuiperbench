
#include "Kuiper_KB_MinReduceDim.h"

__global__
/**
  hoisted when extracting minreduce_dim_fw_f32
*/
static void
__hoisted_minreduce_dim_fw_f32_0(uint32_t m, uint32_t d, float *x, float *y,
                                 uint32_t bm)
{
    if (1024U * blockIdx.x + threadIdx.x < bm) {
        uint32_t ci_ref = 0U;
        float acc_ref = INFINITY;
        for (; ci_ref < d; ci_ref++)
            acc_ref =
                fminf(acc_ref,
                      x[(1024U * blockIdx.x + threadIdx.x) / m * d * m +
                        ci_ref * m + (1024U * blockIdx.x + threadIdx.x) % m]);
        y[1024U * blockIdx.x + threadIdx.x] = acc_ref;
    }
}

void Kuiper_KB_MinReduceDim_minreduce_dim_fw_f32(uint32_t b, uint32_t m,
                                                 uint32_t d, float *x, float *y)
{
    uint32_t bm = b * m;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_minreduce_dim_fw_f32_0,
              bm / 1024U + (uint32_t) (bm % 1024U != 0U), 1024U, 0U, s, m, d, x,
              y, bm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

__global__
/**
  hoisted when extracting minreduce_dim_alloc_f32
*/
static void
__hoisted_minreduce_dim_alloc_f32_0(uint32_t m, uint32_t d, float *x, float *y,
                                    uint32_t bm1)
{
    if (1024U * blockIdx.x + threadIdx.x < bm1) {
        uint32_t ci_ref = 0U;
        float acc_ref = INFINITY;
        for (; ci_ref < d; ci_ref++)
            acc_ref =
                fminf(acc_ref,
                      x[(1024U * blockIdx.x + threadIdx.x) / m * d * m +
                        ci_ref * m + (1024U * blockIdx.x + threadIdx.x) % m]);
        y[1024U * blockIdx.x + threadIdx.x] = acc_ref;
    }
}

float *Kuiper_KB_MinReduceDim_minreduce_dim_alloc_f32(uint32_t b, uint32_t m,
                                                      uint32_t d, float *x)
{
    float *y = (float *) KPR_GPU_ALLOC(sizeof(float), b * m);
    uint32_t bm1 = b * m;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_minreduce_dim_alloc_f32_0,
              bm1 / 1024U + (uint32_t) (bm1 % 1024U != 0U), 1024U, 0U, s, m, d,
              x, y, bm1);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return y;
}
