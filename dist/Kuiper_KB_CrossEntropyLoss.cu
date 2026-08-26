
#include "Kuiper_KB_CrossEntropyLoss.h"

float Kuiper_KB_CrossEntropyLoss_ce_recip_f32(uint32_t b)
{
    return 1.0f / (float) (int64_t) (uint64_t) b;
}

__global__
/**
  hoisted when extracting ce_loss_fw_f32
*/
static void
__hoisted_ce_loss_fw_f32_0(uint32_t c, float *x_, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < c; idx1 += 1024U) {
        float v_ = expf(x_[idx1]);
        acc += v_;
    }
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < 1024U; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa[threadIdx.x] += sa[nextid];
    }
    if (threadIdx.x == 0U)
        out[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting ce_loss_fw_f32
*/
static void
__hoisted_ce_loss_fw_f32_1(uint32_t c, float *scratch, float sum)
{
    if (1024U * blockIdx.x + threadIdx.x < c) {
        float x = scratch[1024U * blockIdx.x + threadIdx.x];
        scratch[1024U * blockIdx.x + threadIdx.x] = x - logf(sum);
    }
}

__global__
/**
  hoisted when extracting ce_loss_fw_f32
*/
static void
__hoisted_ce_loss_fw_f32_2(uint32_t b, float *t_dev, float *out)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx1 = threadIdx.x;
    for (; idx1 < b; idx1 += 1024U)
        acc += t_dev[idx1];
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

float Kuiper_KB_CrossEntropyLoss_ce_loss_fw_f32(uint32_t b, uint32_t c,
                                                float inv_b, float *predictions,
                                                uint32_t *targets)
{
    float *scratch = (float *) KPR_GPU_ALLOC(sizeof(float), c);
    float *t_dev = (float *) KPR_GPU_ALLOC(sizeof(float), b);
    KRML_CHECK_SIZE(sizeof(float), b);
    float *t_host = (float *) KRML_HOST_MALLOC(sizeof(float) * b);
    if (t_host != NULL)
        memset(t_host, 0U, b * sizeof(float));
    uint32_t idx = 0U;
    for (; idx < b; idx++) {
        uint32_t i = idx;
        Kuiper_KB_Compat_Array_gpu_memcpy_device_to_device_(
            (void *) 0U, scratch, 0U, (void *) 0U, predictions, i * c, c,
            (void *) 0U, (void *) 0U, (void *) 0U);
        float *x_ = scratch;
        float *out0 = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
        float *out = out0;
        cudaStream_t s0 = KPR_FRESH_STREAM();
        KPR_SHMEM_FITS(4096U);
        KPR_KCALL(__hoisted_ce_loss_fw_f32_0, 1U, 1024U, 4096U, s0, c, x_, out);
        MUST(cudaStreamSynchronize(s0));
        MUST(cudaStreamDestroy(s0));
        float *local_out = (float *) KRML_HOST_MALLOC(sizeof(float));
        if (local_out != NULL)
            *local_out = 0.0f;
        MUST(cudaMemcpy(local_out, out0, (uint32_t) sizeof(float),
                        cudaMemcpyDeviceToHost));
        float res = *local_out;
        KRML_HOST_FREE(local_out);
        MUST(cudaFree(out0));
        float sum = res;
        cudaStream_t s = KPR_FRESH_STREAM();
        KPR_KCALL(__hoisted_ce_loss_fw_f32_1,
                  c / 1024U + (uint32_t) (c % 1024U != 0U), 1024U, 0U, s, c,
                  scratch, sum);
        MUST(cudaStreamSynchronize(s));
        MUST(cudaStreamDestroy(s));
        float *tmp0 = (float *) KRML_HOST_MALLOC(sizeof(float));
        if (tmp0 != NULL)
            *tmp0 = 0.0f;
        uint32_t *tmp = (uint32_t *) KRML_HOST_CALLOC(1U, sizeof(uint32_t));
        MUST(cudaMemcpy(tmp, targets + i, (uint32_t) sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
        uint32_t x = *tmp;
        KRML_HOST_FREE(tmp);
        MUST(cudaMemcpy(tmp0, scratch + x, (uint32_t) sizeof(float),
                        cudaMemcpyDeviceToHost));
        float x0 = *tmp0;
        KRML_HOST_FREE(tmp0);
        t_host[i] = 0.0f - x0;
    }
    MUST(cudaMemcpy(t_dev, t_host, (uint32_t) sizeof(float) * b,
                    cudaMemcpyHostToDevice));
    float *out = (float *) KPR_GPU_ALLOC(sizeof(float), 1U);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_ce_loss_fw_f32_2, 1U, 1024U, 4096U, s, b, t_dev, out);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    float hout = 0.0f;
    MUST(cudaMemcpy(&hout, out, sizeof(float), cudaMemcpyDeviceToHost));
    MUST(cudaFree(out));
    float m = hout * inv_b;
    KRML_HOST_FREE(t_host);
    MUST(cudaFree(scratch));
    MUST(cudaFree(t_dev));
    return m;
}
