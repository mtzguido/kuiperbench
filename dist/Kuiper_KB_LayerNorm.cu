
#include "Kuiper_KB_LayerNorm.h"

__global__
/**
  hoisted when extracting layer_norm
*/
static void
__hoisted_layer_norm_0(uint32_t n, float *scratch, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < n; idx1 += 1024U)
        acc += scratch[idx1];
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
  hoisted when extracting layer_norm
*/
static void
__hoisted_layer_norm_1(uint32_t n, float *scratch, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < n; idx1 += 1024U) {
        float v = scratch[idx1];
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
  hoisted when extracting layer_norm
*/
static void
__hoisted_layer_norm_2(uint32_t n, float *scratch, float inv,
                       float neg_mean_inv)
{
    if (1024U * blockIdx.x + threadIdx.x < n)
        scratch[1024U * blockIdx.x + threadIdx.x] =
            scratch[1024U * blockIdx.x + threadIdx.x] * inv + neg_mean_inv;
}

__global__
/**
  hoisted when extracting layer_norm
*/
static void
__hoisted_layer_norm_3(uint32_t n, float *gamma, float *scratch)
{
    if (1024U * blockIdx.x + threadIdx.x < n)
        scratch[1024U * blockIdx.x + threadIdx.x] *=
            gamma[1024U * blockIdx.x + threadIdx.x];
}

__global__
/**
  hoisted when extracting layer_norm
*/
static void
__hoisted_layer_norm_4(uint32_t n, float *beta, float *scratch)
{
    if (1024U * blockIdx.x + threadIdx.x < n)
        scratch[1024U * blockIdx.x + threadIdx.x] +=
            beta[1024U * blockIdx.x + threadIdx.x];
}

void Kuiper_KB_LayerNorm_layer_norm(uint32_t b, uint32_t n, float eps,
                                    float inv_n, float *x, float *gamma,
                                    float *beta)
{
    float *scratch = (float *) KPR_GPU_ALLOC(sizeof(float), n);
    uint32_t idx = 0U;
    for (; idx < b; idx++) {
        uint32_t off = idx * n;
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, scratch, 0U, (void *) 0U, x, off, n, (void *) 0U,
            (void *) 0U, (void *) 0U);
        float *out0 = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s0 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_layer_norm_0, 1U, 1024U, 4096U, s0, n, scratch,
                  out0);
        MUST(cudaStreamSynchronize(s0));
        MUST(cudaStreamDestroy(s0));
        float hout = 0.0f;
        MUST(cudaMemcpy(&hout, out0, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out0));
        float mean = hout * inv_n;
        float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_layer_norm_1, 1U, 1024U, 4096U, s1, n, scratch,
                  out);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
        float hout0 = 0.0f;
        MUST(cudaMemcpy(&hout0, out, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out));
        float inv = rsqrtf(hout0 * inv_n - mean * mean + eps);
        float neg_mean_inv = 0.0f - mean * inv;
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_layer_norm_2,
                  n / 1024U + (uint32_t) (n % 1024U != 0U), 1024U, 0U, s, n,
                  scratch, inv, neg_mean_inv);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
        cudaStream_t s2 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_layer_norm_3,
                  n / 1024U + (uint32_t) (n % 1024U != 0U), 1024U, 0U, s2, n,
                  gamma, scratch);
        MUST(cudaStreamSynchronize(s2));
        MUST(cudaStreamDestroy(s2));
        cudaStream_t s3 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_layer_norm_4,
                  n / 1024U + (uint32_t) (n % 1024U != 0U), 1024U, 0U, s3, n,
                  beta, scratch);
        MUST(cudaStreamSynchronize(s3));
        MUST(cudaStreamDestroy(s3));
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, x, off, (void *) 0U, scratch, 0U, n, (void *) 0U,
            (void *) 0U, (void *) 0U);
    }
    MUST(cudaFree(scratch));
}

void Kuiper_KB_LayerNorm_layernorm_fw(uint32_t b, uint32_t n, float eps,
                                      float *x, float *gamma, float *beta)
{
    Kuiper_KB_LayerNorm_layer_norm(
        b, n, eps, 1.0f / (float) (int64_t) (uint64_t) n, x, gamma, beta);
}

void (*Kuiper_KB_LayerNorm_layernorm_fw_f32)(uint32_t x0, uint32_t x1, float x2,
                                             float *x3, float *x4, float *x5) =
    Kuiper_KB_LayerNorm_layernorm_fw;

float *Kuiper_KB_LayerNorm_layernorm4d_alloc_f32(uint32_t b, uint32_t c,
                                                 uint32_t h, uint32_t w,
                                                 double eps, float *x,
                                                 float *gamma, float *beta)
{
    float eps32 = (float) eps;
    uint32_t n = c * h * w;
    uint32_t elems = b * n;
    float *dst = (float *) KPR_GPU_ALLOC(sizeof(float), elems);
    MUST(cudaMemcpy(dst, x, (uint32_t) sizeof(float) * elems,
                    cudaMemcpyDeviceToDevice));
    float *out = dst;
    Kuiper_KB_LayerNorm_layernorm_fw_f32(b, n, eps32, out, gamma, beta);
    return out;
}
