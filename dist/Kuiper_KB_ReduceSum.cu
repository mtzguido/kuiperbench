
#include "Kuiper_KB_ReduceSum.h"

__global__
/**
  hoisted when extracting reduce_sum_fw_f32
*/
static void
__hoisted_reduce_sum_fw_f32_0(uint32_t m, uint32_t d, float *x, float *y)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < d; idx += 1024U)
        acc += x[blockIdx.x / m * d * m + idx * m + blockIdx.x % m];
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

void Kuiper_KB_ReduceSum_reduce_sum_fw_f32(uint32_t b, uint32_t m, uint32_t d,
                                           float *x, float *y)
{
    uint32_t bm = b * m;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_reduce_sum_fw_f32_0, bm, 1024U, 4096U, s, m, d, x, y);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
