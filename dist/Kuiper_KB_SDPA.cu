
#include "Kuiper_KB_SDPA.h"

__global__
/**
  hoisted when extracting smul_fw_f32
*/
static void
__hoisted_smul_fw_f32_0(float c, uint32_t lena, float *a)
{
    if (1024U * blockIdx.x + threadIdx.x < lena)
        a[1024U * blockIdx.x + threadIdx.x] *= c;
}

static void smul_fw_f32(float c, uint32_t lena, float *a)
{
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_smul_fw_f32_0,
              lena / 1024U + (uint32_t) (lena % 1024U != 0U), 1024U, 0U, s1, c,
              lena, a);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
}

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void
__hoisted_row_softmax_rm_f32_0(uint32_t n, float *a, float *maxs, uint32_t nthm)
{
    float *sa1 = (float *) KPR_SHMEM_AT(0U);
    float acc = a[blockIdx.x * n + threadIdx.x];
    uint32_t idx = threadIdx.x + nthm;
    for (; idx < n; idx += nthm)
        acc = fmaxf(acc, a[blockIdx.x * n + idx]);
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < nthm; n1++) {
        uint32_t __anf02 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nthm)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa1[threadIdx.x] = fmaxf(sa1[threadIdx.x], sa1[nextid]);
    }
    if (threadIdx.x == 0U)
        maxs[blockIdx.x] = *sa1;
}

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void
__hoisted_row_softmax_rm_f32_1(uint32_t m, uint32_t n, float *a, float *maxs)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        a[row * n + col] -= maxs[row];
    }
}

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void
__hoisted_row_softmax_rm_f32_2(uint32_t n, uint32_t nth, float *a, float *sums)
{
    float *sa1 = (float *) KPR_SHMEM_AT(0U);
    float acc = 0.0f;
    uint32_t idx = threadIdx.x;
    for (; idx < n; idx += nth) {
        float v_ = expf(a[blockIdx.x * n + idx]);
        acc += v_;
    }
    sa1[threadIdx.x] = acc;
    uint32_t n1 = 0U;
    for (; 1U << (uint32_t) n1 < nth; n1++) {
        uint32_t __anf02 = n1;
        __syncthreads();
        uint32_t nextid = threadIdx.x + (uint32_t) (1U << (uint32_t) __anf02);
        if (nextid < nth)
            if ((threadIdx.x &
                 (uint32_t) (1U << (uint32_t) (__anf02 + 1U)) - 1U) == 0U)
                sa1[threadIdx.x] += sa1[nextid];
    }
    if (threadIdx.x == 0U)
        sums[blockIdx.x] = *sa1;
}

__global__
/**
  hoisted when extracting row_softmax_rm_f32
*/
static void
__hoisted_row_softmax_rm_f32_3(uint32_t m, uint32_t n, float *a, float *sums)
{
    if (1024U * blockIdx.x + threadIdx.x < m * n) {
        uint32_t row = (1024U * blockIdx.x + threadIdx.x) / n;
        uint32_t col = (1024U * blockIdx.x + threadIdx.x) % n;
        float va = sums[row];
        uint32_t ni = row * n + col;
        a[ni] = expf(a[row * n + col]) / va;
    }
}

static void row_softmax_rm_f32(uint32_t m, uint32_t n, uint32_t nth, float *a)
{
    float *maxs = (float *) KPR_GPU_ALLOC(sizeof(float), m);
    float *sums = (float *) KPR_GPU_ALLOC(sizeof(float), m);
    uint32_t nthm = nth <= n ? nth : n;
    cudaStream_t s = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nthm);
    if (4U * nthm >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_row_softmax_rm_f32_0,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  4U * nthm));
    KPR_KCALL(__hoisted_row_softmax_rm_f32_0, m, nthm, 4U * nthm, s, n, a, maxs,
              nthm);
    MUST(cudaStreamSynchronize(s));
    MUST(cudaStreamDestroy(s));
    cudaStream_t s0 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_row_softmax_rm_f32_1,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s0,
              m, n, a, maxs);
    MUST(cudaStreamSynchronize(s0));
    MUST(cudaStreamDestroy(s0));
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_SHMEM_FITS(4U * nth);
    if (4U * nth >= 49152U)
        MUST(cudaFuncSetAttribute(__hoisted_row_softmax_rm_f32_2,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  4U * nth));
    KPR_KCALL(__hoisted_row_softmax_rm_f32_2, m, nth, 4U * nth, s1, n, nth, a,
              sums);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    cudaStream_t s2 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_row_softmax_rm_f32_3,
              m * n / 1024U + (uint32_t) (m * n % 1024U != 0U), 1024U, 0U, s2,
              m, n, a, sums);
    MUST(cudaStreamSynchronize(s2));
    MUST(cudaStreamDestroy(s2));
    MUST(cudaFree(sums));
    MUST(cudaFree(maxs));
}

extern void Kuiper_KB_BatchedGEMM_batched_gemm_f32(uint32_t batch,
                                                   uint32_t rows,
                                                   uint32_t shared,
                                                   uint32_t cols, float *a,
                                                   float *b, float *c);

__global__
/**
  hoisted when extracting sdpa_f32
*/
static void
__hoisted_sdpa_f32_0(uint32_t s, uint32_t d, float *gQ, float *gK, uint32_t bh,
                     float *gScores)
{
    if (1024U * blockIdx.x + threadIdx.x < bh * s * s) {
        uint32_t page = (1024U * blockIdx.x + threadIdx.x) % bh;
        uint32_t rest = (1024U * blockIdx.x + threadIdx.x) / bh;
        uint32_t trow = rest / s;
        uint32_t tcol = rest % s;
        uint32_t k = 0U;
        float sum = 0.0f;
        for (; k < d; k++) {
            uint32_t vk = k;
            sum += gQ[page * s * d + trow * d + vk] *
                   gK[page * d * s + tcol * d + vk];
        }
        gScores[page * s * s + trow * s + tcol] = sum;
    }
}

float *Kuiper_KB_SDPA_sdpa_f32(uint32_t b, uint32_t h, uint32_t s, uint32_t d,
                               float *gQ, float *gK, float *gV)
{
    uint32_t bh = b * h;
    uint32_t bhs = bh * s;
    uint32_t bhsd = bhs * d;
    float *gScores = (float *) KPR_GPU_ALLOC(sizeof(float), bhs * s);
    float *gOut4 = (float *) KPR_GPU_ALLOC(sizeof(float), bhsd);
    float scale = rsqrtf((float) (int64_t) (uint64_t) d);
    cudaStream_t s1 = KPR_FRESH_STREAM();
    KPR_KCALL(__hoisted_sdpa_f32_0,
              bh * s * s / 1024U + (uint32_t) (bh * s * s % 1024U != 0U), 1024U,
              0U, s1, s, d, gQ, gK, bh, gScores);
    MUST(cudaStreamSynchronize(s1));
    MUST(cudaStreamDestroy(s1));
    uint32_t bhs1 = bh * s;
    smul_fw_f32(scale, bhs1 * s, gScores);
    row_softmax_rm_f32(bhs1, s, 1024U, gScores);
    Kuiper_KB_BatchedGEMM_batched_gemm_f32(bh, s, s, d, gScores, gV, gOut4);
    MUST(cudaFree(gScores));
    return gOut4;
}
