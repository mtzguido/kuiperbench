
#include "Kuiper_KB_HingeLoss.h"

__global__
/**
  hoisted when extracting hinge_loss_fw_f32
*/
static void
__hoisted_hinge_loss_fw_f32_0(uint32_t n, float *predictions, float *targets)
{
    if (1024U * blockIdx.x + threadIdx.x < n)
        predictions[1024U * blockIdx.x + threadIdx.x] =
            fmaxf(0.0f, 1.0f - predictions[1024U * blockIdx.x + threadIdx.x] *
                                   targets[1024U * blockIdx.x + threadIdx.x]);
}

__global__
/**
  hoisted when extracting hinge_loss_fw_f32
*/
static void
__hoisted_hinge_loss_fw_f32_1(uint32_t n, float *predictions, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U)
        acc += predictions[idx];
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

float Kuiper_KB_HingeLoss_hinge_loss_fw_f32(uint32_t n, float *predictions,
                                            float *targets)
{
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_hinge_loss_fw_f32_0,
              n / 1024U + (uint32_t) (n % 1024U != 0U), 1024U, 0U, s0, n,
              predictions, targets);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_hinge_loss_fw_f32_1, 1U, 1024U, 4096U, s, n,
              predictions, out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    return hout / (float) (int64_t) (uint64_t) n;
}
