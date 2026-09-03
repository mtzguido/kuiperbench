
#include "Kuiper_KB_LogSoftmaxAlloc.h"

__global__
/**
  hoisted when extracting logsoftmax_alloc_f32
*/
static void
__hoisted_logsoftmax_alloc_f32_0(uint32_t cols, float *output, float *sums)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < cols; idx += 1024U) {
        float v_ = expf(output[blockIdx.x * cols + idx]);
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
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting logsoftmax_alloc_f32
*/
static void
__hoisted_logsoftmax_alloc_f32_1(uint32_t rows, uint32_t cols, float *output,
                                 float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % cols;
        float vb = output[row * cols + col];
        uint32_t ni = row * cols + col;
        output[ni] = vb - logf(sums[row]);
    }
}

float *Kuiper_KB_LogSoftmaxAlloc_logsoftmax_alloc_f32(uint32_t rows,
                                                      uint32_t cols,
                                                      float *input)
{
    float *_return;
    bool _return1 = false;
    float *dst = (float *) KPR_GPU_ALLOC(sizeof(float), rows * cols);
    MUST(cudaMemcpy(dst, input, (uint32_t) sizeof(float) * rows * cols,
                    cudaMemcpyDeviceToDevice));
    float *output = dst;
    float *sums = (float *) KPR_GPU_ALLOC(sizeof(float), rows);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_logsoftmax_alloc_f32_0, rows, 1024U, 4096U, s1, cols,
              output, sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s10 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_logsoftmax_alloc_f32_1,
              rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
              1024U, 0U, s10, rows, cols, output, sums);
    MUST(cudaStreamSynchronize(s10));
    MUST(cudaStreamDestroy(s10));
    MUST(cudaFree(sums));
    _return = output;
    _return1 = true;
    return _return;
}

__global__
/**
  hoisted when extracting logsoftmax_alloc_f64
*/
static void
__hoisted_logsoftmax_alloc_f64_0(uint32_t cols, double *output, double *sums)
{
    double *sa = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < cols; idx += 1024U) {
        double v_ = exp(output[blockIdx.x * cols + idx]);
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
        sums[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting logsoftmax_alloc_f64
*/
static void
__hoisted_logsoftmax_alloc_f64_1(uint32_t rows, uint32_t cols, double *output,
                                 double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % cols;
        double vb = output[row * cols + col];
        uint32_t ni = row * cols + col;
        output[ni] = vb - log(sums[row]);
    }
}

double *Kuiper_KB_LogSoftmaxAlloc_logsoftmax_alloc_f64(uint32_t rows,
                                                       uint32_t cols,
                                                       double *input)
{
    double *_return;
    bool _return1 = false;
    double *dst = (double *) KPR_GPU_ALLOC(sizeof(double), rows * cols);
    MUST(cudaMemcpy(dst, input, (uint32_t) sizeof(double) * rows * cols,
                    cudaMemcpyDeviceToDevice));
    double *output = dst;
    double *sums = (double *) KPR_GPU_ALLOC(sizeof(double), rows);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_logsoftmax_alloc_f64_0, rows, 1024U, 8192U, s1, cols,
              output, sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s10 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_logsoftmax_alloc_f64_1,
              rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
              1024U, 0U, s10, rows, cols, output, sums);
    MUST(cudaStreamSynchronize(s10));
    MUST(cudaStreamDestroy(s10));
    MUST(cudaFree(sums));
    _return = output;
    _return1 = true;
    return _return;
}
