
#include "Kuiper_KB_MeanVarNorm.h"

__global__
/**
  hoisted when extracting mean_var_norm
*/
static void
__hoisted_mean_var_norm_0(uint32_t d, float *scratch, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < d; idx1 += 1024U)
        acc += scratch[idx1];
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
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
  hoisted when extracting mean_var_norm
*/
static void
__hoisted_mean_var_norm_1(uint32_t d, float *scratch, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < d; idx1 += 1024U) {
        float v = scratch[idx1];
        acc += v * v;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf01 = n;
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
  hoisted when extracting mean_var_norm
*/
static void
__hoisted_mean_var_norm_2(uint32_t d, float *scratch, float inv,
                          float neg_mean_inv)
{
    if (1024U * blockIdx.x + threadIdx.x < d)
        scratch[1024U * blockIdx.x + threadIdx.x] =
            scratch[1024U * blockIdx.x + threadIdx.x] * inv + neg_mean_inv;
}

void Kuiper_KB_MeanVarNorm_mean_var_norm(uint32_t b, uint32_t d, float eps,
                                         float inv_d, float *x)
{
    float *scratch = (float *) KPR_GPU_ALLOC(sizeof(float), d);
    uint32_t idx = 0U;
    for (; idx < b; idx++) {
        uint32_t off = idx * d;
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, scratch, 0U, (void *) 0U, x, off, d, (void *) 0U,
            (void *) 0U, (void *) 0U);
        float *out0 = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s0 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_mean_var_norm_0, 1U, 1024U, 4096U, s0, d, scratch,
                  out0);
        MUST(cudaStreamSynchronize(s0));
        MUST(cudaStreamDestroy(s0));
        float hout = 0.0f;
        MUST(cudaMemcpy(&hout, out0, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out0));
        float mean = hout * inv_d;
        float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_mean_var_norm_1, 1U, 1024U, 4096U, s1, d, scratch,
                  out);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
        float hout0 = 0.0f;
        MUST(cudaMemcpy(&hout0, out, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out));
        float inv = rsqrtf(hout0 * inv_d - mean * mean + eps);
        float neg_mean_inv = 0.0f - mean * inv;
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_mean_var_norm_2,
                  d / 1024U + (uint32_t) (d % 1024U != 0U), 1024U, 0U, s, d,
                  scratch, inv, neg_mean_inv);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, x, off, (void *) 0U, scratch, 0U, d, (void *) 0U,
            (void *) 0U, (void *) 0U);
    }
    MUST(cudaFree(scratch));
}

void Kuiper_KB_MeanVarNorm_mean_var_norm_fw(uint32_t b, uint32_t d, float eps,
                                            float *x)
{
    Kuiper_KB_MeanVarNorm_mean_var_norm(
        b, d, eps, 1.0f / (float) (int64_t) (uint64_t) d, x);
}

void (*Kuiper_KB_MeanVarNorm_mean_var_norm_fw_f32)(uint32_t x0, uint32_t x1,
                                                   float x2, float *x3) =
    Kuiper_KB_MeanVarNorm_mean_var_norm_fw;
