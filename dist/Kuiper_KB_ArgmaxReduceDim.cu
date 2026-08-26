
#include "Kuiper_KB_ArgmaxReduceDim.h"

__global__
/**
  hoisted when extracting argmaxreduce_dim_fw_f32
*/
static void
__hoisted_argmaxreduce_dim_fw_f32_0(uint32_t m, uint32_t d, float *x,
                                    int64_t *y, uint32_t bm)
{
    if (1024U * blockIdx.x + threadIdx.x < bm) {
        uint32_t ci_ref = 0U;
        uint32_t bi_ref = 0U;
        float bv_ref = 0.0f - INFINITY;
        for (; ci_ref < d; ci_ref++) {
            uint32_t ci_v = ci_ref;
            float v = x[(1024U * blockIdx.x + threadIdx.x) / m * d * m +
                        ci_v * m + (1024U * blockIdx.x + threadIdx.x) % m];
            if (bv_ref < v) {
                bv_ref = v;
                bi_ref = ci_v;
            }
        }
        y[1024U * blockIdx.x + threadIdx.x] = (int64_t) (uint32_t) bi_ref;
    }
}

void Kuiper_KB_ArgmaxReduceDim_argmaxreduce_dim_fw_f32(uint32_t b, uint32_t m,
                                                       uint32_t d, float *x,
                                                       int64_t *y)
{
    uint32_t bm = b * m;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_argmaxreduce_dim_fw_f32_0,
              bm / 1024U + (uint32_t) (bm % 1024U != 0U), 1024U, 0U, s, m, d, x,
              y, bm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}
