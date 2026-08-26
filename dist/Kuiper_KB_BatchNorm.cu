
#include "Kuiper_KB_BatchNorm.h"

__global__
/**
  hoisted when extracting batchnorm_fw
*/
static void
__hoisted_batchnorm_fw_0(uint32_t c, uint32_t hw, uint32_t nhw, float *x,
                         uint32_t i, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < nhw; idx1 += 1024U) {
        uint32_t vidx = idx1;
        acc += x[vidx / hw * c * hw + i * hw + vidx % hw];
    }
    sa[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf01 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) - 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting batchnorm_fw
*/
static void
__hoisted_batchnorm_fw_1(uint32_t c, uint32_t hw, uint32_t nhw, float *x,
                         uint32_t i, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < nhw; idx1 += 1024U) {
        uint32_t vidx = idx1;
        float v = x[vidx / hw * c * hw + i * hw + vidx % hw];
        acc += v * v;
    }
    sa[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf01 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) - 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa;
}

__global__
/**
  hoisted when extracting batchnorm_fw
*/
static void
__hoisted_batchnorm_fw_2(uint32_t c, uint32_t hw, uint32_t nhw, float *x,
                         uint32_t i, float inv, float neg_mean_inv)
{
    if (1024U * blockIdx.x + threadIdx.x < nhw)
        x[(1024U * blockIdx.x + threadIdx.x) / hw * c * hw + i * hw +
          (1024U * blockIdx.x + threadIdx.x) % hw] =
            x[(1024U * blockIdx.x + threadIdx.x) / hw * c * hw + i * hw +
              (1024U * blockIdx.x + threadIdx.x) % hw] *
                inv +
            neg_mean_inv;
}

__global__
/**
  hoisted when extracting batchnorm_fw
*/
static void
__hoisted_batchnorm_fw_3(uint32_t c, uint32_t hw, uint32_t nhw, float *x,
                         uint32_t i, float g_c, float b_c)
{
    if (1024U * blockIdx.x + threadIdx.x < nhw)
        x[(1024U * blockIdx.x + threadIdx.x) / hw * c * hw + i * hw +
          (1024U * blockIdx.x + threadIdx.x) % hw] =
            x[(1024U * blockIdx.x + threadIdx.x) / hw * c * hw + i * hw +
              (1024U * blockIdx.x + threadIdx.x) % hw] *
                g_c +
            b_c;
}

void Kuiper_KB_BatchNorm_batchnorm_fw(uint32_t c, uint32_t hw, uint32_t nhw,
                                      float eps, float *x, float *gamma,
                                      float *beta)
{
    float inv_n = 1.0f / (float) (int64_t) (uint64_t) nhw;
    uint32_t idx = 0U;
    for (; idx < c; idx++) {
        uint32_t i = idx;
        float *ca0 = (float *) KRML_HOST_MALLOC(sizeof(float));
        if (ca0 != NULL)
            *ca0 = 0.0f;
        MUST(cudaMemcpy(ca0, gamma + i, (uint32_t) sizeof(float),
                        cudaMemcpyDeviceToHost));
        float x10 = *ca0;
        KRML_HOST_FREE(ca0);
        float g_c = x10;
        float *ca = (float *) KRML_HOST_MALLOC(sizeof(float));
        if (ca != NULL)
            *ca = 0.0f;
        MUST(cudaMemcpy(ca, beta + i, (uint32_t) sizeof(float),
                        cudaMemcpyDeviceToHost));
        float x1 = *ca;
        KRML_HOST_FREE(ca);
        float b_c = x1;
        float *out0 = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s0 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_batchnorm_fw_0, 1U, 1024U, 4096U, s0, c, hw, nhw, x,
                  i, out0);
        MUST(cudaStreamSynchronize(s0));
        MUST(cudaStreamDestroy(s0));
        float hout = 0.0f;
        MUST(cudaMemcpy(&hout, out0, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out0));
        float mean = hout * inv_n;
        float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_batchnorm_fw_1, 1U, 1024U, 4096U, s1, c, hw, nhw, x,
                  i, out);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
        float hout0 = 0.0f;
        MUST(cudaMemcpy(&hout0, out, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out));
        float inv = rsqrtf(hout0 * inv_n - mean * mean + eps);
        float neg_mean_inv = 0.0f - mean * inv;
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_batchnorm_fw_2,
                  nhw / 1024U + (uint32_t) (nhw % 1024U != 0U), 1024U, 0U, s, c,
                  hw, nhw, x, i, inv, neg_mean_inv);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
        cudaStream_t s2 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_batchnorm_fw_3,
                  nhw / 1024U + (uint32_t) (nhw % 1024U != 0U), 1024U, 0U, s2,
                  c, hw, nhw, x, i, g_c, b_c);
        MUST(cudaStreamSynchronize(s2));
        MUST(cudaStreamDestroy(s2));
    }
}

void (*Kuiper_KB_BatchNorm_batchnorm_fw_f32)(
    uint32_t x0, uint32_t x1, uint32_t x2, float x3, float *x4, float *x5,
    float *x6) = Kuiper_KB_BatchNorm_batchnorm_fw;
