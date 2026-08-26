
#include "Kuiper_KB_RMSNorm.h"

__global__
/**
  hoisted when extracting rmsnorm_fw
*/
static void
__hoisted_rmsnorm_fw_0(uint32_t hw, uint32_t c, float *x, uint32_t bhw,
                       float *sum_sq)
{
    if (1024U * blockIdx.x + threadIdx.x < bhw) {
        uint32_t ci_ref = 0U;
        float acc_ref = 0.0f;
        for (; ci_ref < c; ci_ref++) {
            float v = x[(1024U * blockIdx.x + threadIdx.x) / hw * c * hw +
                        ci_ref * hw + (1024U * blockIdx.x + threadIdx.x) % hw];
            acc_ref += v * v;
        }
        sum_sq[1024U * blockIdx.x + threadIdx.x] = acc_ref;
    }
}

__global__
/**
  hoisted when extracting rmsnorm_fw
*/
static void
__hoisted_rmsnorm_fw_1(float eps, float inv_c, uint32_t bhw, float *sum_sq)
{
    if (1024U * blockIdx.x + threadIdx.x < bhw)
        sum_sq[1024U * blockIdx.x + threadIdx.x] =
            rsqrtf(sum_sq[1024U * blockIdx.x + threadIdx.x] * inv_c + eps);
}

__global__
/**
  hoisted when extracting rmsnorm_fw
*/
static void
__hoisted_rmsnorm_fw_2(uint32_t hw, uint32_t c, float *x, uint32_t bhw,
                       float *sum_sq)
{
    if (1024U * blockIdx.x + threadIdx.x < bhw * c) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / c;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % c;
        x[row / hw * c * hw + col * hw + row % hw] *= sum_sq[row];
    }
}

void Kuiper_KB_RMSNorm_rmsnorm_fw(uint32_t b, uint32_t hw, uint32_t c,
                                  float eps, float *x)
{
    float inv_c = 1.0f / (float) (int64_t) (uint64_t) c;
    uint32_t bhw = b * hw;
    float *sum_sq = (float *) KPR_GPU_ALLOC(sizeof(float), bhw);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rmsnorm_fw_0,
              bhw / 1024U + (uint32_t) (bhw % 1024U != 0U), 1024U, 0U, s, hw, c,
              x, bhw, sum_sq);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rmsnorm_fw_1,
              bhw / 1024U + (uint32_t) (bhw % 1024U != 0U), 1024U, 0U, s0, eps,
              inv_c, bhw, sum_sq);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_rmsnorm_fw_2,
              bhw * c / 1024U + (uint32_t) (bhw * c % 1024U != 0U), 1024U, 0U,
              s1, hw, c, x, bhw, sum_sq);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    MUST(cudaFree(sum_sq));
}

void (*Kuiper_KB_RMSNorm_rmsnorm_fw_f32)(uint32_t x0, uint32_t x1, uint32_t x2,
                                         float x3, float *x4) =
    Kuiper_KB_RMSNorm_rmsnorm_fw;
