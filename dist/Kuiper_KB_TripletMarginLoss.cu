
#include "Kuiper_KB_TripletMarginLoss.h"

__global__
/**
  hoisted when extracting triplet_fw_f32
*/
static void
__hoisted_triplet_fw_f32_0(uint32_t d, float eps, float *scratch_a,
                           float *scratch_b)
{
    if (1024U * blockIdx.x + threadIdx.x < d) {
        float d1 = scratch_a[1024U * blockIdx.x + threadIdx.x] -
                   scratch_b[1024U * blockIdx.x + threadIdx.x] + eps;
        scratch_a[1024U * blockIdx.x + threadIdx.x] = d1 * d1;
    }
}

__global__
/**
  hoisted when extracting triplet_fw_f32
*/
static void
__hoisted_triplet_fw_f32_1(uint32_t d, float *scratch_a, float *out)
{
    float *sa1 = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < d; idx1 += 1024U)
        acc += scratch_a[idx1];
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf01 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) - 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa1;
}

__global__
/**
  hoisted when extracting triplet_fw_f32
*/
static void
__hoisted_triplet_fw_f32_2(uint32_t d, float eps, float *scratch_a,
                           float *scratch_b)
{
    if (1024U * blockIdx.x + threadIdx.x < d) {
        float d1 = scratch_a[1024U * blockIdx.x + threadIdx.x] -
                   scratch_b[1024U * blockIdx.x + threadIdx.x] + eps;
        scratch_a[1024U * blockIdx.x + threadIdx.x] = d1 * d1;
    }
}

__global__
/**
  hoisted when extracting triplet_fw_f32
*/
static void
__hoisted_triplet_fw_f32_3(uint32_t d, float *scratch_a, float *out)
{
    float *sa1 = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < d; idx1 += 1024U)
        acc += scratch_a[idx1];
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf01 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) - 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa1;
}

__global__
/**
  hoisted when extracting triplet_fw_f32
*/
static void
__hoisted_triplet_fw_f32_4(uint32_t b, float *t_dev, float *out)
{
    float *sa1 = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < b; idx1 += 1024U)
        acc += t_dev[idx1];
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf01 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf01);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf01 + 1U)) - 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        *out = *sa1;
}

float Kuiper_KB_TripletMarginLoss_triplet_fw_f32(uint32_t b, uint32_t d,
                                                 float margin, float eps,
                                                 float *a, float *p, float *n)
{
    float inv_b = 1.0f / (float) (int64_t) (uint64_t) b;
    float *scratch_a = (float *) KPR_GPU_ALLOC(sizeof(float), d);
    float *scratch_b = (float *) KPR_GPU_ALLOC(sizeof(float), d);
    float *t_dev = (float *) KPR_GPU_ALLOC(sizeof(float), b);
    KRML_CHECK_SIZE(sizeof(float), b);
    float *t_host = (float *) KRML_HOST_MALLOC(sizeof(float) * b);
    if (t_host != NULL)
        memset(t_host, 0U, b * sizeof(float));
    uint32_t idx = 0U;
    for (; idx < b; idx++) {
        uint32_t i = idx;
        uint32_t off = i * d;
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, scratch_a, 0U, (void *) 0U, a, off, d, (void *) 0U,
            (void *) 0U, (void *) 0U);
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, scratch_b, 0U, (void *) 0U, p, off, d, (void *) 0U,
            (void *) 0U, (void *) 0U);
        cudaStream_t s0 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_triplet_fw_f32_0,
                  d / 1024U + (uint32_t) (d % 1024U != 0U), 1024U, 0U, s0, d,
                  eps, scratch_a, scratch_b);
        MUST(cudaStreamSynchronize(s0));
        MUST(cudaStreamDestroy(s0));
        float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_triplet_fw_f32_1, 1U, 1024U, 4096U, s1, d,
                  scratch_a, out);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
        float hout = 0.0f;
        MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out));
        float d_ap_r = sqrtf(hout);
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, scratch_a, 0U, (void *) 0U, a, off, d, (void *) 0U,
            (void *) 0U, (void *) 0U);
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, scratch_b, 0U, (void *) 0U, n, off, d, (void *) 0U,
            (void *) 0U, (void *) 0U);
        cudaStream_t s2 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_triplet_fw_f32_2,
                  d / 1024U + (uint32_t) (d % 1024U != 0U), 1024U, 0U, s2, d,
                  eps, scratch_a, scratch_b);
        MUST(cudaStreamSynchronize(s2));
        MUST(cudaStreamDestroy(s2));
        float *out0 = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_triplet_fw_f32_3, 1U, 1024U, 4096U, s, d, scratch_a,
                  out0);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
        float hout0 = 0.0f;
        MUST(cudaMemcpy(&hout0, out0, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out0));
        t_host[i] = fmaxf(0.0f, d_ap_r - sqrtf(hout0) + margin);
    }
    MUST(cudaMemcpy(t_dev, t_host, (uint32_t) sizeof(float) * b,
                    cudaMemcpyHostToDevice));
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_triplet_fw_f32_4, 1U, 1024U, 4096U, s, b, t_dev, out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float m = hout * inv_b;
    KRML_HOST_FREE(t_host);
    MUST(cudaFree(scratch_a));
    MUST(cudaFree(scratch_b));
    MUST(cudaFree(t_dev));
    return m;
}
