
#include "Kuiper_KB_Frobenius.h"

__global__
/**
  hoisted when extracting frobenius_fw_f32
*/
static void
__hoisted_frobenius_fw_f32_0(uint32_t lena, float *a, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < lena; idx += 1024U) {
        float v = a[idx];
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
  hoisted when extracting frobenius_fw_f32
*/
static void
__hoisted_frobenius_fw_f32_1(uint32_t lena, float *a, float inv_norm)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *= inv_norm;
}

void Kuiper_KB_Frobenius_frobenius_fw_f32(uint32_t lena, float *a)
{
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_frobenius_fw_f32_0, 1U, 1024U, 4096U, s0, lena, a, out);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float inv_norm = rsqrtf(hout);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_frobenius_fw_f32_1,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s,
              lena, a, inv_norm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
}

float *Kuiper_KB_Frobenius_frobenius_alloc_f32(uint32_t lena, float *a)
{
    float *dst = (float *) KPR_GPU_ALLOC(sizeof(float), lena);
    MUST(cudaMemcpy(dst, a, (uint32_t) sizeof(float) * lena,
                    cudaMemcpyDeviceToDevice));
    float *out = dst;
    Kuiper_KB_Frobenius_frobenius_fw_f32(lena, out);
    return out;
}
