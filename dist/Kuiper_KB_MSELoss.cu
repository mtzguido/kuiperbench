
#include "Kuiper_KB_MSELoss.h"

__global__
/**
  hoisted when extracting mse_scalar_out_f32
*/
static void
__hoisted_mse_scalar_out_f32_0(float x, float *out)
{
    if (1024U * blockIdx.x + threadIdx.x < 1U)
        out[1024U * blockIdx.x + threadIdx.x] = x;
}

float *Kuiper_KB_MSELoss_mse_scalar_out_f32(float x)
{
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_mse_scalar_out_f32_0, 1U, 1024U, 0U, s, x, out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    return out;
}

__global__
/**
  hoisted when extracting mse_loss_fw_f32
*/
static void
__hoisted_mse_loss_fw_f32_0(uint32_t n, float *targets, float *scratch)
{
    if (1024U * blockIdx.x + threadIdx.x < n) {
        float d = scratch[1024U * blockIdx.x + threadIdx.x] -
                  targets[1024U * blockIdx.x + threadIdx.x];
        scratch[1024U * blockIdx.x + threadIdx.x] = d * d;
    }
}

__global__
/**
  hoisted when extracting mse_loss_fw_f32
*/
static void
__hoisted_mse_loss_fw_f32_1(uint32_t n, float *scratch, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U)
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

float *Kuiper_KB_MSELoss_mse_loss_fw_f32(uint32_t n, float *predictions,
                                         float *targets)
{
    float *scratch = (float *) KPR_GPU_ALLOC(sizeof(float), n);
    MUST(cudaMemcpy(scratch, predictions, (uint32_t) sizeof(float) * n,
                    cudaMemcpyDeviceToDevice));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_mse_loss_fw_f32_0,
              n / 1024U + (uint32_t) (n % 1024U != 0U), 1024U, 0U, s0, n,
              targets, scratch);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_mse_loss_fw_f32_1, 1U, 1024U, 4096U, s, n, scratch,
              out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float res = hout / (float) (int64_t) (uint64_t) n;
    MUST(cudaFree(scratch));
    return Kuiper_KB_MSELoss_mse_scalar_out_f32(res);
}
