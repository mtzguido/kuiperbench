
#include "Kuiper_KB_SoftmaxAlloc.h"

__global__
/**
  hoisted when extracting softmax_alloc_f32
*/
static void
__hoisted_softmax_alloc_f32_0(uint32_t cols, float *output, float *maxs,
                              uint32_t nthm)
{
    float *sa = (float *) KPR_SHMEM_AT(0U);
    float acc = output[blockIdx.x * cols + threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < cols; idx += nthm)
        acc = fmaxf(acc, output[blockIdx.x * cols + idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_alloc_f32
*/
static void
__hoisted_softmax_alloc_f32_1(uint32_t rows, uint32_t cols, float *output,
                              float *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % cols;
        output[row * cols + col] -= maxs[row];
    }
}

__global__
/**
  hoisted when extracting softmax_alloc_f32
*/
static void
__hoisted_softmax_alloc_f32_2(uint32_t cols, float *output, float *sums)
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
  hoisted when extracting softmax_alloc_f32
*/
static void
__hoisted_softmax_alloc_f32_3(uint32_t rows, uint32_t cols, float *output,
                              float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % cols;
        float va = sums[row];
        uint32_t ni = row * cols + col;
        output[ni] = expf(output[row * cols + col]) / va;
    }
}

float *Kuiper_KB_SoftmaxAlloc_softmax_alloc_f32(uint32_t rows, uint32_t cols,
                                                float *input)
{
    float *_return;
    bool _return1 = false;
    float *dst = (float *) KPR_GPU_ALLOC(sizeof(float), rows * cols);
    MUST(cudaMemcpy(dst, input, (uint32_t) sizeof(float) * rows * cols,
                    cudaMemcpyDeviceToDevice));
    float *output = dst;
    float *maxs = (float *) KPR_GPU_ALLOC(sizeof(float), rows);
    float *sums = (float *) KPR_GPU_ALLOC(sizeof(float), rows);
    uint32_t nthm = 1024U <= cols ? 1024U : cols;
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nthm);
    if (4U * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_alloc_f32_0,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  4U * nthm));
    KPR_KCALL(__hoisted_softmax_alloc_f32_0, rows, nthm, 4U * nthm, s1, cols,
              output, maxs, nthm);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s10 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_alloc_f32_1,
              rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
              1024U, 0U, s10, rows, cols, output, maxs);
    MUST(cudaStreamSynchronize(s10));
    MUST(cudaStreamDestroy(s10));
    cudaStream_t s11 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_softmax_alloc_f32_2, rows, 1024U, 4096U, s11, cols,
              output, sums);
    MUST(cudaStreamSynchronize(s11));
    MUST(cudaStreamDestroy(s11));
    cudaStream_t s12 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_alloc_f32_3,
              rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
              1024U, 0U, s12, rows, cols, output, sums);
    MUST(cudaStreamSynchronize(s12));
    MUST(cudaStreamDestroy(s12));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    _return = output;
    _return1 = true;
    return _return;
}

__global__
/**
  hoisted when extracting softmax_alloc_f64
*/
static void
__hoisted_softmax_alloc_f64_0(uint32_t cols, double *output, double *maxs,
                              uint32_t nthm)
{
    double *sa = (double *) KPR_SHMEM_AT(0U);
    double acc = output[blockIdx.x * cols + threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < cols; idx += nthm)
        acc = fmax(acc, output[blockIdx.x * cols + idx]);
    sa[threadIdx.x] = acc;
    uint32_t n = 0U;
    for (; 1U << (uint32_t) n < nthm; n++) {
        uint32_t __anf02 = n;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa;
}

__global__
/**
  hoisted when extracting softmax_alloc_f64
*/
static void
__hoisted_softmax_alloc_f64_1(uint32_t rows, uint32_t cols, double *output,
                              double *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % cols;
        output[row * cols + col] -= maxs[row];
    }
}

__global__
/**
  hoisted when extracting softmax_alloc_f64
*/
static void
__hoisted_softmax_alloc_f64_2(uint32_t cols, double *output, double *sums)
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
  hoisted when extracting softmax_alloc_f64
*/
static void
__hoisted_softmax_alloc_f64_3(uint32_t rows, uint32_t cols, double *output,
                              double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < rows * cols) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / cols;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % cols;
        double va = sums[row];
        uint32_t ni = row * cols + col;
        output[ni] = exp(output[row * cols + col]) / va;
    }
}

double *Kuiper_KB_SoftmaxAlloc_softmax_alloc_f64(uint32_t rows, uint32_t cols,
                                                 double *input)
{
    double *_return;
    bool _return1 = false;
    double *dst = (double *) KPR_GPU_ALLOC(sizeof(double), rows * cols);
    MUST(cudaMemcpy(dst, input, (uint32_t) sizeof(double) * rows * cols,
                    cudaMemcpyDeviceToDevice));
    double *output = dst;
    double *maxs = (double *) KPR_GPU_ALLOC(sizeof(double), rows);
    double *sums = (double *) KPR_GPU_ALLOC(sizeof(double), rows);
    uint32_t nthm = 1024U <= cols ? 1024U : cols;
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8U * nthm);
    if (8U * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_softmax_alloc_f64_0,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  8U * nthm));
    KPR_KCALL(__hoisted_softmax_alloc_f64_0, rows, nthm, 8U * nthm, s1, cols,
              output, maxs, nthm);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s10 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_alloc_f64_1,
              rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
              1024U, 0U, s10, rows, cols, output, maxs);
    MUST(cudaStreamSynchronize(s10));
    MUST(cudaStreamDestroy(s10));
    cudaStream_t s11 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_softmax_alloc_f64_2, rows, 1024U, 8192U, s11, cols,
              output, sums);
    MUST(cudaStreamSynchronize(s11));
    MUST(cudaStreamDestroy(s11));
    cudaStream_t s12 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_softmax_alloc_f64_3,
              rows * cols / 1024U + (uint32_t) (rows * cols % 1024U != 0U),
              1024U, 0U, s12, rows, cols, output, sums);
    MUST(cudaStreamSynchronize(s12));
    MUST(cudaStreamDestroy(s12));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
    _return = output;
    _return1 = true;
    return _return;
}
