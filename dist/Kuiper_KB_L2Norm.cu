
#include "Kuiper_KB_L2Norm.h"

__global__
/**
  hoisted when extracting l2norm_fw_f32
*/
static void
__hoisted_l2norm_fw_f32_0(uint32_t d, float *scratch, float *out)
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
  hoisted when extracting l2norm_fw_f32
*/
static void
__hoisted_l2norm_fw_f32_1(uint32_t d, float *scratch, float inv)
{
    if (1024U * blockIdx.x + threadIdx.x < d)
        scratch[1024U * blockIdx.x + threadIdx.x] *= inv;
}

void Kuiper_KB_L2Norm_l2norm_fw_f32(uint32_t b, uint32_t d, float *x)
{
    float *scratch = (float *) KPR_GPU_ALLOC(sizeof(float), d);
    uint32_t idx = 0U;
    for (; idx < b; idx++) {
        uint32_t off = idx * d;
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, scratch, 0U, (void *) 0U, x, off, d, (void *) 0U,
            (void *) 0U, (void *) 0U);
        float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        cudaStream_t s10 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_l2norm_fw_f32_0, 1U, 1024U, 4096U, s10, d, scratch,
                  out);
        MUST(cudaStreamSynchronize(s10));
        MUST(cudaStreamDestroy(s10));
        float hout = 0.0f;
        MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
        MUST(cudaFree(out));
        float inv = rsqrtf(hout);
        cudaStream_t s1 = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_l2norm_fw_f32_1,
                  d / 1024U + (uint32_t) (d % 1024U != 0U), 1024U, 0U, s1, d,
                  scratch, inv);
        MUST(cudaStreamSynchronize(s1));
        MUST(cudaStreamDestroy(s1));
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, x, off, (void *) 0U, scratch, 0U, d, (void *) 0U,
            (void *) 0U, (void *) 0U);
    }
    MUST(cudaFree(scratch));
}
