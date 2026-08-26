
#include "Kuiper_KB_MaxReduceDim.h"

__global__
/**
  hoisted when extracting maxreduce_dim_fw_f32
*/
static void
__hoisted_maxreduce_dim_fw_f32_0(uint32_t m, uint32_t d, float *x, float *y,
                                 uint32_t nthm)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = x[blockIdx.x / m * d * m + threadIdx.x * m + blockIdx.x % m];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < d; idx += nthm)
        acc = fmaxf(acc, x[blockIdx.x / m * d * m + idx * m + blockIdx.x % m]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        y[blockIdx.x] = *sa;
}

void Kuiper_KB_MaxReduceDim_maxreduce_dim_fw_f32(uint32_t b, uint32_t m,
                                                 uint32_t d, float *x, float *y)
{
    uint32_t bm = b * m;
    uint32_t nthm = 1024U <= d ? 1024U : d;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nthm);
    if (4U * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_maxreduce_dim_fw_f32_0,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  4U * nthm));
    KPR_KCALL(__hoisted_maxreduce_dim_fw_f32_0, bm, nthm, 4U * nthm, s, m, d, x,
              y, nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
