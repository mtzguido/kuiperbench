
#include "Kuiper_KB_HingeLoss.h"

__global__
/**
  hoisted when extracting hinge_loss_broadcast_f32
*/
static void
__hoisted_hinge_loss_broadcast_f32_0(uint32_t b, uint32_t n, float *targets,
                                     float *scratch)
{
    if (1024U * blockIdx.x + threadIdx.x < n * b) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / b;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % b;
        uint32_t ni = col * n + row;
        scratch[ni] = fmaxf(0.0f, 1.0f - scratch[col * n + row] * targets[row]);
    }
}

__global__
/**
  hoisted when extracting hinge_loss_broadcast_f32
*/
static void
__hoisted_hinge_loss_broadcast_f32_1(uint32_t elems, float *scratch, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < elems; idx += 1024U)
        acc += scratch[idx];
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
  hoisted when extracting hinge_loss_broadcast_f32
*/
static void
__hoisted_hinge_loss_broadcast_f32_2(float mean, float *out)
{
    if (1024U * blockIdx.x + threadIdx.x < 1U)
        out[1024U * blockIdx.x + threadIdx.x] = mean;
}

float *Kuiper_KB_HingeLoss_hinge_loss_broadcast_f32(uint32_t b, uint32_t n,
                                                    float *predictions,
                                                    float *targets)
{
    uint32_t elems = b * n;
    float *scratch = (float *) KPR_GPU_ALLOC(sizeof(float), elems);
    MUST(cudaMemcpy(scratch, predictions, (uint32_t) sizeof(float) * b * n,
                    cudaMemcpyDeviceToDevice));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_hinge_loss_broadcast_f32_0,
              n * b / 1024U + (uint32_t) (n * b % 1024U != 0U), 1024U, 0U, s0,
              b, n, targets, scratch);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    float *out0 = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_hinge_loss_broadcast_f32_1, 1U, 1024U, 4096U, s1, elems,
              scratch, out0);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out0, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out0));
    float sum = hout;
    MUST(cudaFree(scratch));
    float mean = sum / (float) (int64_t) (uint64_t) elems;
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_hinge_loss_broadcast_f32_2, 1U, 1024U, 0U, s, mean,
              out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return out;
}
