
#include "Klas_RowLogSoftmax.h"

__global__
/**
  hoisted when extracting row_log_softmax_rm_f32
*/
static void
__hoisted_row_log_softmax_rm_f32_0(uint32_t n, float *a, float *sums)
{
    float *sa1 = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U) {
        float v_ = expf(a[blockIdx.x * n + idx]);
        acc += v_;
    }
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf02 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa1;
}

__global__
/**
  hoisted when extracting row_log_softmax_rm_f32
*/
static void
__hoisted_row_log_softmax_rm_f32_1(uint32_t m, uint32_t n, float *a,
                                   float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        float vb = a[row * n + col];
        uint32_t ni = row * n + col;
        a[ni] = vb - logf(sums[row]);
    }
}

void Klas_RowLogSoftmax_row_log_softmax_rm_f32(uint32_t m, uint32_t n, float *a)
{
    float *sums = (float *) KPR_GPU_ALLOC(sizeof(float), m);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4096U);
    KPR_KCALL(__hoisted_row_log_softmax_rm_f32_0, m, 1024U, 4096U, s, n, a,
              sums);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_row_log_softmax_rm_f32_1,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s0,
              m, n, a, sums);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    MUST(cudaFree(sums));
}

__global__
/**
  hoisted when extracting row_log_softmax_rm_f64
*/
static void
__hoisted_row_log_softmax_rm_f64_0(uint32_t n, double *a, double *sums)
{
    double *sa1 = (double *) KPR_SHMEM_AT(0U);
    double acc = 0.0;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += 1024U) {
        double v_ = exp(a[blockIdx.x * n + idx]);
        acc += v_;
    }
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < 1024U; n1++) {
        uint32_t __anf02 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < 1024U)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa1;
}

__global__
/**
  hoisted when extracting row_log_softmax_rm_f64
*/
static void
__hoisted_row_log_softmax_rm_f64_1(uint32_t m, uint32_t n, double *a,
                                   double *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        double vb = a[row * n + col];
        uint32_t ni = row * n + col;
        a[ni] = vb - log(sums[row]);
    }
}

void Klas_RowLogSoftmax_row_log_softmax_rm_f64(uint32_t m, uint32_t n,
                                               double *a)
{
    double *sums = (double *) KPR_GPU_ALLOC(sizeof(double), m);
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(8192U);
    KPR_KCALL(__hoisted_row_log_softmax_rm_f64_0, m, 1024U, 8192U, s, n, a,
              sums);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_row_log_softmax_rm_f64_1,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s0,
              m, n, a, sums);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    MUST(cudaFree(sums));
}
